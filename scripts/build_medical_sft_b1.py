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


# 改进2: 问题模板规范化 — 将 esp32-ai 机械模板重写为自然问句
# 模式: "根据临床指南，【X】的内容要点有哪些" / "根据临床指南，X的内容要点有哪些"
TEMPLATE_RE = re.compile(r'^根据临床指南[,，]\s*(?:【([^】]+)】|(.+?))的?内容要点有哪些$')
# 常见的"XX的临床诊疗要点是什么" -> 保持 (已是自然问句)
# 子节模式: "根据临床指南，（二）病原治疗的内容要点有哪些" -> 提取括号内小节名


def normalize_question(q):
    """将模板化问题重写为自然问句."""
    m = TEMPLATE_RE.match(q)
    if not m:
        return q
    section = (m.group(1) or m.group(2) or '').strip()
    # 去掉括号序号如 （二）
    section = re.sub(r'^[（(][一二三四五六七八九十\d]+[)）]', '', section).strip()
    # 去掉可能的 "的诊疗要点"/"的治疗" 等尾巴残留
    section = re.sub(r'的?(临床表现|诊断要点|治疗原则及方案|鉴别诊断|流行病学|实验室检查|预后|预防|概述)$', r'\1', section)
    if not section:
        return q
    # 当 section 本身就是标准节名时, 直接用通用问法
    std_sections = ['临床表现', '诊断要点', '治疗原则及方案', '鉴别诊断', '实验室检查', '流行病学', '预后', '预防', '概述']
    if section in std_sections:
        return f'{section}是什么？' if section in ('临床表现', '流行病学', '概述') else f'{section}有哪些？'
    # 映射为自然问句
    mapping = {
        '临床表现': f'{section}的临床表现是什么？',
        '诊断要点': f'{section}的诊断要点有哪些？',
        '治疗原则及方案': f'{section}的治疗原则是什么？',
        '鉴别诊断': f'{section}如何鉴别诊断？',
        '实验室检查': f'{section}需要做哪些实验室检查？',
        '流行病学': f'{section}的流行病学特点是什么？',
        '预后': f'{section}的预后如何？',
        '预防': f'{section}如何预防？',
        '概述': f'{section}是什么？',
    }
    for key, q_new in mapping.items():
        if key in section:
            return q_new
    return f'{section}的诊疗要点有哪些？'


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
            q = normalize_question(q)  # 改进2: 模板 → 自然问句
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
