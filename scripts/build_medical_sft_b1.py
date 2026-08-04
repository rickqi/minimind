#!/usr/bin/env python3
"""
管线 B1: format_data.jsonl → minimind SFT 格式 (直接转换, 零成本)

输入: esp32-ai/data_v4/kb/format_data.jsonl (11,000 条可读医学 QA)
输出: dataset/sft_medical_b1.jsonl
      {"conversations": [{"role":"user","content":q},{"role":"assistant","content":a}]}

用法:
  python scripts/build_medical_sft_b1.py
"""

import argparse
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, '..', 'esp32-ai', 'data_v4', 'kb', 'format_data.jsonl')

WS_RE = re.compile(r'\s+')


def clean_text(s):
    """基础清洗: 去空白压缩/HTML残留/超长截断."""
    s = re.sub(r'<[^>]+>', ' ', s)
    s = WS_RE.sub(' ', s).strip()
    return s


def main():
    ap = argparse.ArgumentParser(description='管线B1: format_data.jsonl → minimind SFT')
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b1.jsonl'))
    ap.add_argument('--report', default=os.path.join(PROJECT_ROOT, 'out', 'medical_sft_b1_report.json'))
    ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()

    t0 = time.time()
    if not os.path.exists(args.src):
        print(f'[ERR] 源不存在: {args.src}')
        sys.exit(1)

    n_total = n_kept = n_dup = 0
    seen = set()
    lens = []
    with open(args.src, 'r', encoding='utf-8') as fin, open(args.out, 'w', encoding='utf-8') as fout:
        for line in fin:
            n_total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = clean_text(d.get('question', ''))
            a = clean_text(d.get('answer', ''))
            if len(q) < 5 or len(a) < 5:
                continue
            if len(q) > args.max_len or len(a) > args.max_len:
                continue
            # 精确去重 (question, answer)
            key = (q, a)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            sample = {'conversations': [
                {'role': 'user', 'content': q},
                {'role': 'assistant', 'content': a},
            ]}
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            n_kept += 1
            lens.append(len(q) + len(a))

    report = {
        'total': n_total,
        'kept': n_kept,
        'dup_removed': n_dup,
        'avg_qal_len': sum(lens) / max(len(lens), 1),
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[out] {args.out}: {n_kept} samples')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
