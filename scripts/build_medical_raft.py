#!/usr/bin/env python3
"""
RAFT 数据构建 (移植 esp32-ai build_raft.py 思路, 适配 minimind ChatML)

esp32-ai 核心发现: 仅注入证据不训练, 小模型会忽略证据 (重复/幻觉)。
RAFT 用 (证据+问题 -> 答案) 训练, 让模型学会"引用证据"。

格式 (minimind SFT):
  {"conversations": [
    {"role":"system","content":"你是一个医学助手，请根据提供的参考资料准确回答问题。"},
    {"role":"user","content":"参考资料：\n{evidence1}\n{evidence2}\n\n问题：{question}"},
    {"role":"assistant","content":"{answer}"}
  ]}

证据 = 同条目答案的前 60 字符 (自接地, esp32-ai P2 发现: 仅答案, 短截断)
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KB_PATH = os.path.join(PROJECT_ROOT, '..', 'esp32-ai', 'data_v4', 'kb', 'format_data.jsonl')


def load_kb():
    entries = []
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            entries.append((d.get('question', ''), d.get('answer', ''), d.get('label', '')))
    return entries


NON_MEDICAL_KW = ['健康管理', '理赔', '产品条款', '销售', '消保']
# 真正的保险/理赔内容关键词 (内容级过滤)
INSURANCE_CONTENT_KW = ['理赔', '保险', '保额', '赔付', '免责', '条款', '退保', '续保',
                        '报案', '报销', '保费', '承保', '除外责任', '投保']


def is_medical_label(label):
    """判断 KB 条目是否为医学 (排除保险/健康管理域)."""
    return not any(k in label for k in NON_MEDICAL_KW)


def is_medical_content(question):
    """内容级过滤: 保留医学问答, 剔除保险/理赔内容 (优化3修正).
    健康管理标签 84% 是真实医学内容, 不能仅按标签过滤."""
    if any(k in question for k in INSURANCE_CONTENT_KW):
        return False
    # 医学内容判定: 含临床诊疗/诊断/治疗/症状等医学语义
    med_kw = ['诊疗', '诊断', '治疗', '症状', '疾病', '患者', '药物', '检查',
              '症', '病', '炎', '痛', '综合征', '分型', '分期', '护理', '康复']
    return any(k in question for k in med_kw)


def build_raft_data(out_path, max_samples=8000, seed=42, no_evidence_ratio=0.3,
                    negative_ratio=0.15, med_only=False):
    """构建 RAFT 数据.
    no_evidence_ratio: 无证据样本比例 (保留模型内在知识问答能力, 防 RAFT 遗忘).
    negative_ratio: 负样本 (拒答式) 比例 — 给无关证据, 教模型拒绝 (esp32-ai 未实现的优化1).
    med_only: 仅用医学标签条目 (优化3: 剔除保险/健康管理域).
    证据格式 (v4):
      E1 = 正确答案 answer[:60]  (自接地, 模拟检索 Top-1 命中)
      E2 = 随机其他条目 answer[:60]  (干扰项, 模拟真实 Top-2 检索的无关文档)
    负样本: E1 = 随机无关条目 answer[:60] (与问题无关), 目标 = 拒答.
    """
    import random
    rng = random.Random(seed)

    entries = load_kb()
    if med_only:
        # 组合过滤: 剔除保险标签 + 保留医学内容 (健康管理标签 84% 是医学, 需内容级判断)
        entries = [e for e in entries
                   if is_medical_label(e[2]) or is_medical_content(e[0])]
        print(f'[med_only] 组合过滤后 {len(entries)} 条 (医学标签+内容)',
              flush=True)
    # 候选: 答案足够长的条目
    cands = []
    for i, (q, a, label) in enumerate(entries):
        if a and len(a) >= 40:
            cands.append(i)
    rng.shuffle(cands)
    cands = cands[:max_samples]
    # 干扰池: 所有答案 (E2 随机采样源)
    all_answers = [a[:60] for q, a, label in entries if a]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    n_ne = 0
    n_neg = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, i in enumerate(cands):
            q, a, label = entries[i]
            # 按 no_evidence_ratio 概率生成无证据样本 (防 RAFT 遗忘内在知识)
            if rng.random() < no_evidence_ratio:
                ev = ''
                n_ne += 1
            elif rng.random() < negative_ratio:
                # 负样本 (拒答式): 无关证据 -> 拒绝回答 (esp32-ai 未实现, 修复盲引)
                e1 = rng.choice(all_answers)  # 与问题无关的证据
                e2 = rng.choice(all_answers)
                ev = f'{e1}\n{e2}'
                n_neg += 1
                target = '根据提供的参考资料，无法确定该问题的答案。参考资料中未提及相关信息。'
            else:
                e1 = a[:60]                     # 正确答案前缀 (Top-1 命中)
                e2 = rng.choice(all_answers)    # 随机干扰项 (模拟 Top-2 无关文档)
                ev = f'{e1}\n{e2}'
                target = a
            if ev:
                user_content = f'参考资料：\n{ev}\n\n问题：{q}'
            else:
                user_content = q
                target = a
            conv = [
                {'role': 'system', 'content': '你是一个医学助手，请根据提供的参考资料准确回答问题。'},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': target},
            ]
            f.write(json.dumps({'conversations': conv}, ensure_ascii=False) + '\n')
            n += 1
    print(f'[done] {out_path}: {n} RAFT samples (no-ev {n_ne}, neg {n_neg}, '
          f'{100*n_neg/max(n,1):.0f}% negative)', flush=True)
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='构建 RAFT 微调数据')
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_raft.jsonl'))
    ap.add_argument('--max-samples', type=int, default=8000)
    ap.add_argument('--no-evidence-ratio', type=float, default=0.3,
                    help='无证据样本比例 (防 RAFT 遗忘内在知识, 默认 0.3)')
    ap.add_argument('--negative-ratio', type=float, default=0.15,
                    help='负样本 (拒答式) 比例 — 无关证据教模型拒绝 (默认 0.15)')
    ap.add_argument('--med-only', action='store_true',
                    help='仅用医学标签条目 (剔除保险/健康管理域, 优化3)')
    args = ap.parse_args()
    build_raft_data(args.out, args.max_samples, no_evidence_ratio=args.no_evidence_ratio,
                    negative_ratio=args.negative_ratio, med_only=args.med_only)
