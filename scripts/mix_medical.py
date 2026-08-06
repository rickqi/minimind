#!/usr/bin/env python3
"""
混合策略: 医疗数据 + 通用数据 → 混合训练集

策略 (基于 esp32-ai V4 经验调整):
  Pretrain: pretrain_medical.jsonl : pretrain_t2t_mini.jsonl = 1 : 2  (医学 1/3)
  SFT:      sft_medical_b1 + b2 : sft_t2t_mini.jsonl = 1 : 3        (医学 1/4)

输出:
  - dataset/pretrain_mixed.jsonl
  - dataset/sft_medical_mixed.jsonl
  - out/medical_mix_report.json

用法:
  python scripts/mix_medical.py
  python scripts/mix_medical.py --pretrain-ratio 1 --pretrain-base 2
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            n += 1
    return n


def sample_lines(src, n, seed=42):
    """从 src 均匀采样 n 行 (单次流式读, 蓄水池采样避免重复读大文件)."""
    if not os.path.exists(src):
        return []
    rng = random.Random(seed)
    # 蓄水池采样: 保证均匀且单次读取
    reservoir = []
    with open(src, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    return reservoir


def main():
    ap = argparse.ArgumentParser(description='混合医疗+通用数据')
    ap.add_argument('--pretrain-med', default=os.path.join(PROJECT_ROOT, 'dataset', 'pretrain_medical.jsonl'))
    ap.add_argument('--pretrain-base', default=os.path.join(PROJECT_ROOT, 'dataset', 'pretrain_t2t_mini.jsonl'))
    ap.add_argument('--pretrain-ratio', type=int, default=1, help='医学份数')
    ap.add_argument('--pretrain-base-ratio', type=int, default=2, help='通用份数')
    ap.add_argument('--sft-med-b1', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b1.jsonl'))
    ap.add_argument('--sft-med-b2', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b2.jsonl'))
    ap.add_argument('--sft-base', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_t2t_mini.jsonl'))
    ap.add_argument('--sft-ratio', type=int, default=1, help='医学SFT份数')
    ap.add_argument('--sft-base-ratio', type=int, default=3, help='通用SFT份数')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    # ---- Pretrain 混合 ----
    n_med = count_lines(args.pretrain_med)
    target_base = int(n_med * args.pretrain_base_ratio / args.pretrain_ratio)
    print(f'[pretrain] medical {n_med} -> sample base {target_base} from {args.pretrain_base}', flush=True)
    med_lines = sample_lines(args.pretrain_med, n_med, args.seed)
    base_lines = sample_lines(args.pretrain_base, target_base, args.seed + 1)
    print(f'[pretrain] med {len(med_lines)} + base {len(base_lines)}', flush=True)
    all_lines = med_lines + base_lines
    random.Random(args.seed + 2).shuffle(all_lines)
    out_p = os.path.join(PROJECT_ROOT, 'dataset', 'pretrain_mixed.jsonl')
    with open(out_p, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line)

    # ---- SFT 混合 ----
    n_b1 = count_lines(args.sft_med_b1)
    n_b2 = count_lines(args.sft_med_b2)
    n_sft_med = n_b1 + n_b2
    target_sft_base = int(n_sft_med * args.sft_base_ratio / args.sft_ratio)
    print(f'[sft] medical {n_sft_med} (b1 {n_b1} + b2 {n_b2}) -> sample base {target_sft_base}', flush=True)
    b1_lines = sample_lines(args.sft_med_b1, n_b1, args.seed)
    b2_lines = sample_lines(args.sft_med_b2, n_b2, args.seed)
    sft_base_lines = sample_lines(args.sft_base, target_sft_base, args.seed + 3)
    print(f'[sft] b1 {len(b1_lines)} + b2 {len(b2_lines)} + base {len(sft_base_lines)}', flush=True)
    sft_all = b1_lines + b2_lines + sft_base_lines
    random.Random(args.seed + 4).shuffle(sft_all)
    out_s = os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_mixed.jsonl')
    with open(out_s, 'w', encoding='utf-8') as f:
        for line in sft_all:
            f.write(line)

    report = {
        'pretrain': {'medical': n_med, 'base': target_base, 'total': len(all_lines)},
        'sft': {'medical_b1': n_b1, 'medical_b2': n_b2, 'base': target_sft_base, 'total': len(sft_all)},
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.join(PROJECT_ROOT, 'out'), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, 'out', 'medical_mix_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'[out] {out_p} ({os.path.getsize(out_p)/1e6:.1f}MB)')
    print(f'[out] {out_s} ({os.path.getsize(out_s)/1e6:.1f}MB)')


if __name__ == '__main__':
    main()
