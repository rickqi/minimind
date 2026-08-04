#!/usr/bin/env python3
"""
管线 B2: DeepSeek V4 Flash 医学 SFT QA 生成

步骤:
  1. 从 临床诊疗指南全集/*.pdf.md 提取疾病-节块 (第X节 疾病名 + ## 【临床表现|诊断要点|...】)
  2. 用 DeepSeek V4 Flash (1M 上下文) 批量生成 QA, 单次调用喂入一个疾病全文
  3. JSON 校验 + 忠实度检查 + 断点续跑 (缓存已生成结果)

用法:
  python scripts/build_medical_sft_b2.py --api-key sk-xxx
  python scripts/build_medical_sft_b2.py --api-key sk-xxx --max-diseases 5   # 试运行
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE_DIR = r'D:\docs\raw\临床诊疗指南全集'
if not os.path.isdir(GUIDE_DIR):
    GUIDE_DIR = '/mnt/d/docs/raw/临床诊疗指南全集'  # WSL 兼容

# 【】分节锚点
SECTION_RE = re.compile(r'^##\s*【([^】]+)】\s*$')
# 章标题 (任何 # 级别的 "第X章")
CHAPTER_RE = re.compile(r'^#+\s*第[一二三四五六七八九十百]+章')
# 节标题 (疾病名): ## 第X节 疾病名 (正文格式)
SECTION_TITLE_RE = re.compile(r'^##\s*第[一二三四五六七八九十百]+节\s*([^\s·]{2,20})')
# 清洗用
PAGE_RE = re.compile(r'<!--\s*Page\s*\d+\s*-->')
HTML_RE = re.compile(r'<[^>]+>')
IMG_RE = re.compile(r'!\[\]\([^)]*\)')
LATEX_RE = re.compile(r'\$[^$]*\$')


def clean(text):
    text = PAGE_RE.sub('\n', text)
    text = IMG_RE.sub(' ', text)
    text = HTML_RE.sub(' ', text)
    text = LATEX_RE.sub(' ', text)
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
        if re.match(r'^[#\s\d\.\-—·]+$', s):
            continue
        if 'PDF电子书基地' in s or 'QQ' in s and len(s) < 30:
            continue
        lines.append(s)
    return '\n'.join(lines)


def parse_diseases(md_path):
    """提取 (disease, {section: text}) 结构."""
    with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    diseases = OrderedDict()  # disease_name -> {section: [lines]}
    cur_disease, cur_section = None, None
    for line in lines:
        s = line.strip()
        m_ch = CHAPTER_RE.match(s)
        if m_ch:
            cur_disease, cur_section = None, None
            continue
        m_sec = SECTION_TITLE_RE.match(s)
        if m_sec:
            cur_disease = m_sec.group(1).strip()
            cur_section = None
            diseases.setdefault(cur_disease, OrderedDict())
            continue
        m_sect = SECTION_RE.match(s)
        if m_sect:
            cur_section = m_sect.group(1)
            if cur_disease is None:
                continue
            diseases[cur_disease].setdefault(cur_section, [])
            continue
        if cur_disease is not None and cur_section is not None:
            if s and not s.startswith('#'):
                diseases[cur_disease][cur_section].append(s)

    # 过滤: 只保留至少有 1 个标准分节且正文 ≥ 50 字符的疾病
    valid = OrderedDict()
    for disease, sections in diseases.items():
        total = sum(len(ls) for ls in sections.values())
        std = [k for k in sections if k in ('概述', '临床表现', '诊断要点', '治疗原则及方案', '实验室检查', '鉴别诊断', '流行病学')]
        if std and total >= 30:
            valid[disease] = {k: '\n'.join(v) for k, v in sections.items() if v}
    return valid


def build_disease_prompt(disease, sections, max_chars=30000):
    """构建单个疾病的完整 prompt (V4 Flash 1M 上下文可容纳, 但保护单次请求)."""
    parts = [f'# 疾病: {disease}', '']
    total = 0
    for sec, text in sections.items():
        if text.strip():
            block = f'## {sec}\n{text.strip()}\n'
            if total + len(block) > max_chars:
                break  # 截断超长疾病, 保留前面分节
            parts.append(block)
            total += len(block)
    return '\n'.join(parts)


def generate_qa(client, disease, sections, max_qas=6):
    """调用 V4 Flash 生成 QA 对."""
    prompt = build_disease_prompt(disease, sections)
    system = (
        '你是资深医学数据标注员。根据提供的临床指南章节内容，生成高质量中文医学问答对。\n'
        '要求：\n'
        '1. 问题覆盖【临床表现】【诊断要点】【治疗原则及方案】等核心内容\n'
        '2. 答案严格忠实于原文，禁止编造原文没有的信息\n'
        '3. 答案简洁准确，50-200字，可适当归纳\n'
        '4. 只输出 JSON 数组，不要任何其他文字\n'
        f'5. 生成 {max_qas} 条问答对，格式: [{{"question": "...", "answer": "..."}}]'
    )
    resp = client.chat.completions.create(
        model='deepseek-v4-flash',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=4000,
        temperature=0.3,
        response_format={'type': 'json_object'},
    )
    content = resp.choices[0].message.content
    # 解析 JSON (可能是 list 或 {"data": [...]})
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 数组
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get('data') or data.get('qa') or data.get('pairs') or []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict):
            q = str(item.get('question') or item.get('q') or '').strip()
            a = str(item.get('answer') or item.get('a') or '').strip()
            if len(q) >= 5 and len(a) >= 20:
                out.append((q, a))
    return out


def main():
    ap = argparse.ArgumentParser(description='管线B2: V4 Flash 医学 QA 生成')
    ap.add_argument('--api-key', required=True, help='DeepSeek API key')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'sft_medical_b2.jsonl'))
    ap.add_argument('--cache', default=os.path.join(PROJECT_ROOT, 'out', 'b2_cache.json'))
    ap.add_argument('--report', default=os.path.join(PROJECT_ROOT, 'out', 'medical_sft_b2_report.json'))
    ap.add_argument('--max-diseases', type=int, default=0, help='处理疾病数上限 (0=全部)')
    ap.add_argument('--max-qas', type=int, default=6, help='每疾病生成 QA 数')
    ap.add_argument('--model', default='deepseek-v4-flash')
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=120.0, max_retries=2)

    # 1. 提取所有疾病
    all_diseases = OrderedDict()
    for md in glob.glob(os.path.join(GUIDE_DIR, '*.pdf.md')):
        name = os.path.basename(md)
        if '_ocr' in name:
            continue
        try:
            diseases = parse_diseases(md)
        except Exception as e:
            print(f'[warn] {name}: {e}')
            continue
        for d, sections in diseases.items():
            all_diseases.setdefault(d, sections)
    print(f'[parse] 提取 {len(all_diseases)} 个疾病')

    if args.max_diseases:
        all_diseases = OrderedDict(list(all_diseases.items())[:args.max_diseases])

    # 2. 加载缓存 (断点续跑)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    print(f'[cache] 已有 {len(cache)} 个疾病缓存')

    # 3. 批量生成
    t0 = time.time()
    n_qa = 0
    n_fail = 0
    for i, (disease, sections) in enumerate(all_diseases.items()):
        if disease in cache:
            n_qa += len(cache[disease])
            continue
        try:
            qas = generate_qa(client, disease, sections, args.max_qas)
        except Exception as e:
            print(f'[fail] {disease}: {str(e)[:100]}')
            n_fail += 1
            continue
        cache[disease] = qas
        n_qa += len(qas)
        # 每疾病后写缓存 (断点续跑)
        with open(args.cache, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        if (i + 1) % 5 == 0 or i == len(all_diseases) - 1:
            eta = (time.time() - t0) / (i + 1) * (len(all_diseases) - i - 1)
            print(f'[{i+1}/{len(all_diseases)}] {disease}: {len(qas)} QA | '
                  f'total {n_qa} | fail {n_fail} | ETA {eta/60:.0f}min')

    # 4. 写输出
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for disease, qas in cache.items():
            for q, a in qas:
                sample = {'conversations': [
                    {'role': 'user', 'content': q},
                    {'role': 'assistant', 'content': a},
                ]}
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                n_written += 1
    print(f'\n[out] {args.out}: {n_written} samples')

    # 5. 报告
    report = {
        'diseases_total': len(all_diseases),
        'qa_total': n_written,
        'fail': n_fail,
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
