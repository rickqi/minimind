#!/usr/bin/env python3
"""
生成纯医学 SFT 数据集: B1(过滤低质量) + B2 合并, 无通用数据稀释.

过滤规则 (B1):
  - 问题含原文片段: 以编号开头或含冒号的长问句 (残缺模板残留)
  - 问题超长 (>120 字符)
  - 非问句 (陈述句, 无 ?/？/是什么/有哪些/如何)

输出: dataset/sft_medical_pure.jsonl
用法:
  python scripts/build_medical_sft_pure.py
"""

import argparse
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 残缺问题模式: 以编号开头 (原文片段残留)
NUM_PREFIX_RE = re.compile(r'^[（(]?[一二三四五六七八九十\d]+[)）]?\s*\d*[、.．]?\s*[A-Za-z]')
# 含": "或"："的长问句 (疑似截断的原文片段)
COLON_LONG_RE = re.compile(r'[：:].{20,}')


def is_low_quality(q, a, max_q_len=120):
    """判断 B1 样本是否低质量 (残缺/模板化)."""
    if len(q) > max_q_len:
        return True
    if NUM_PREFIX_RE.match(q):
        return True
    if '？' not in q and '?' not in q and '有哪些' not in q and '是什么' not in q and '如何' not in q:
        return True  # 非问句
    if COLON_LONG_RE.search(q):
        return True  # 含长冒号片段 (截断残留)
    return False


def main():
    ap = argparse.ArgumentParser(description='生成纯医学 SFT 数据集 (B1过滤+B2)')
    ap.add_argument('--b1', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b1.jsonl'))
    ap.add_argument('--b2', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b2.jsonl'))
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_pure.jsonl'))
    ap.add_argument('--report', default=os.path.join(PROJECT_ROOT, 'out', 'medical_sft_pure_report.json'))
    args = ap.parse_args()

    t0 = time.time()
    n_b1_in = n_b1_kept = n_b1_filtered = 0
    n_b2 = 0
    n_total = 0
    seen = set()
    with open(args.out, 'w', encoding='utf-8') as fout:
        # B1 过滤
        with open(args.b1, 'r', encoding='utf-8') as f:
            for line in f:
                n_b1_in += 1
                d = json.loads(line)
                q = d['conversations'][0]['content']
                a = d['conversations'][1]['content']
                if is_low_quality(q, a):
                    n_b1_filtered += 1
                    continue
                key = (q, a)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(line)
                n_b1_kept += 1
                n_total += 1
        # B2 全保留
        with open(args.b2, 'r', encoding='utf-8') as f:
            for line in f:
                n_b2 += 1
                d = json.loads(line)
                key = (d['conversations'][0]['content'], d['conversations'][1]['content'])
                if key in seen:
                    continue
                seen.add(key)
                fout.write(line)
                n_total += 1

    report = {
        'b1_in': n_b1_in,
        'b1_filtered': n_b1_filtered,
        'b1_kept': n_b1_kept,
        'b2': n_b2,
        'total': n_total,
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[out] {args.out}: {n_total} samples')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
