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


def build_raft_data(out_path, max_samples=8000, seed=42):
    import random
    rng = random.Random(seed)

    entries = load_kb()
    # esp32-ai build_raft.py 核心: 证据 = 同条目答案切片 (自接地, 无需检索)
    # E1 = answer[:60], E2 = answer[60:120]
    cands = []
    for i, (q, a, label) in enumerate(entries):
        if a and len(a) >= 40:
            cands.append(i)
    rng.shuffle(cands)
    cands = cands[:max_samples]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in cands:
            q, a, label = entries[i]
            e1, e2 = a[:60], a[60:120]
            ev = f'{e1}\n{e2}'
            conv = [
                {'role': 'system', 'content': '你是一个医学助手，请根据提供的参考资料准确回答问题。'},
                {'role': 'user', 'content': f'参考资料：\n{ev}\n\n问题：{q}'},
                {'role': 'assistant', 'content': a},
            ]
            f.write(json.dumps({'conversations': conv}, ensure_ascii=False) + '\n')
            n += 1
    print(f'[done] {out_path}: {n} RAFT samples', flush=True)
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='构建 RAFT 微调数据')
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_raft.jsonl'))
    ap.add_argument('--max-samples', type=int, default=8000)
    args = ap.parse_args()
    build_raft_data(args.out, args.max_samples)
