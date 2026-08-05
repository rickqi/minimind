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


def build_raft_data(out_path, max_samples=8000, seed=42, no_evidence_ratio=0.3):
    """构建 RAFT 数据.
    no_evidence_ratio: 无证据样本比例 (保留模型内在知识问答能力, 防 RAFT 遗忘).
    证据格式 (v3, 修复 E2 分布):
      E1 = 正确答案 answer[:60]  (自接地, 模拟检索 Top-1 命中)
      E2 = 随机其他条目 answer[:60]  (干扰项, 模拟真实 Top-2 检索的无关文档)
    修复前 E2 = 同答案 answer[60:120] (续接), 与推理分布不匹配.
    """
    import random
    rng = random.Random(seed)

    entries = load_kb()
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
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, i in enumerate(cands):
            q, a, label = entries[i]
            # 按 no_evidence_ratio 概率生成无证据样本 (防 RAFT 遗忘内在知识)
            if rng.random() < no_evidence_ratio:
                ev = ''
                n_ne += 1
            else:
                e1 = a[:60]                     # 正确答案前缀 (Top-1 命中)
                e2 = rng.choice(all_answers)    # 随机干扰项 (模拟 Top-2 无关文档)
                ev = f'{e1}\n{e2}'
            if ev:
                user_content = f'参考资料：\n{ev}\n\n问题：{q}'
            else:
                user_content = q
            conv = [
                {'role': 'system', 'content': '你是一个医学助手，请根据提供的参考资料准确回答问题。'},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': a},
            ]
            f.write(json.dumps({'conversations': conv}, ensure_ascii=False) + '\n')
            n += 1
    print(f'[done] {out_path}: {n} RAFT samples ({n_ne} no-evidence, {100*n_ne/max(n,1):.0f}%)', flush=True)
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='构建 RAFT 微调数据')
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_raft.jsonl'))
    ap.add_argument('--max-samples', type=int, default=8000)
    ap.add_argument('--no-evidence-ratio', type=float, default=0.3,
                    help='无证据样本比例 (防 RAFT 遗忘内在知识, 默认 0.3)')
    args = ap.parse_args()
    build_raft_data(args.out, args.max_samples, no_evidence_ratio=args.no_evidence_ratio)
