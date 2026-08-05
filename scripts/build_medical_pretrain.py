#!/usr/bin/env python3
"""
管线 A: 医疗 Pretrain 语料构建 (清洗 + 分块 + MinHash 去重 + 质量报告)

输入三源:
  - D:\codes\esp32-ai\data_v4\corpus.txt          (348MB 已清洗医学文本)
  - D:\docs\raw\medica\**\*.pdf.md                (444 医疗指南 md)
  - D:\docs\raw\临床诊疗指南全集\**\*.pdf.md       (76 临床分册 md)

输出:
  - dataset/pretrain_medical.jsonl                {"text": "..."}
  - out/medical_pretrain_report.json              质量报告 (可重复执行对比)

用法:
  python scripts/build_medical_pretrain.py
  python scripts/build_medical_pretrain.py --out dataset/pretrain_medical.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

import jieba  # noqa: F401  (预留; 去重实际用字符 n-gram, 见 minhash_of)
from datasketch import MinHash, MinHashLSH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 清洗规则
# ---------------------------------------------------------------------------
# 水印/损坏文件黑名单 (L4 中 4 个 watermark-only 失败文件)
BLOCKLIST_SUBSTR = [
    '临床技术操作规范 呼吸病学分册',
    '临床技术操作规范 麻醉学分册',
    '临床技术操作规范 手外科分册',
    '临床技术操作规范 烧伤分册',
    'PDF电子书基地', 'dayo1982', 'QQ461573687', 'QQ1779903665',
    '本人可以帮助你找到你要的PDF', '代找', '每本100%都带可跳转',
    # 广告/版权页短语 (改进1: 分析发现的残留噪声)
    '帮助了上万人', '带书签索引', '电子书代找', 'pdf代找', '电子书基地',
    '因寻找和后期制作pdf', '仅收取代找费用', '版权纠纷', '本人只提供代找',
    '如因PDF产生的版权', '可以联系我QQ', '网上有很多PDF',
    # 版权/出版页
    '版权所有, 翻印必究', '版权所有，翻印必究', '出版时间', '印数', '书号',
    'ISBN', '在版编目',
]

# YAML frontmatter: --- 到 ---
FRONTMATTER_RE = re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL | re.MULTILINE)
# 页码标记
PAGE_RE = re.compile(r'<!--\s*Page\s*\d+\s*-->')
# HTML 标签/表格
HTML_RE = re.compile(r'<[^>]+>')
# 图片占位
IMG_RE = re.compile(r'!\[\]\([^)]*\)')
# LaTeX 数学片段
LATEX_RE = re.compile(r'\$[^$]*\$')
# 连续空白
WS_RE = re.compile(r'[ \t\u3000]+')
MULTI_NL_RE = re.compile(r'\n{3,}')
# 目录页码行 (如 "## 1. 导言 ....... 4")
TOC_RE = re.compile(r'^#{1,4}\s.*[\.·]{4,}\s*\d+\s*$', re.MULTILINE)
# 纯数字/页码行
NUM_ONLY_RE = re.compile(r'^\s*[\d\s\-—.]{1,10}\s*$')


def is_noise_line(line):
    """判断单行是否噪声."""
    s = line.strip()
    if not s:
        return True
    if NUM_ONLY_RE.match(s):
        return True
    for kw in BLOCKLIST_SUBSTR:
        if kw in s:
            return True
    return False


def clean_markdown(text):
    """清洗单个 md 文本: 去 frontmatter/HTML/LaTeX/样板/噪声行."""
    text = FRONTMATTER_RE.sub('', text)
    text = PAGE_RE.sub('\n', text)
    text = IMG_RE.sub(' ', text)
    text = HTML_RE.sub(' ', text)
    text = LATEX_RE.sub(' ', text)
    text = TOC_RE.sub('\n', text)
    # 逐行过滤
    lines = text.split('\n')
    kept = []
    for line in lines:
        if is_noise_line(line):
            continue
        # 去 markdown 标题符号 (保留文字)
        s = re.sub(r'^#{1,6}\s*', '', line).strip()
        s = WS_RE.sub(' ', s)
        if s:
            kept.append(s)
    out = '\n'.join(kept)
    out = MULTI_NL_RE.sub('\n\n', out)
    return out.strip()


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------
def _hard_split(text, max_len):
    """硬切分: 按字符窗口强制切分 (兜底, 用于 >2000 字符的超长块)."""
    out = []
    for i in range(0, len(text), max_len):
        out.append(text[i:i + max_len])
    return out


def chunk_text(text, min_len=80, max_len=1024, overlap=0, hard_max=2000):
    """按段落分块, 目标 512-1024 字符. 短段合并, 超长段按句子切, 最终兜底硬切."""
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, cur = [], []

    def flush():
        nonlocal cur
        if len(cur) >= min_len:
            chunks.append(cur)
        cur = ''

    for p in paras:
        if len(p) > max_len:
            flush()
            # 超长段落按句子切 (中文句号/分号/换行)
            sentences = re.split(r'(?<=[。；;！？!?])\s*', p)
            sub = ''
            for s in sentences:
                if len(sub) + len(s) > max_len and sub:
                    if len(sub) >= min_len:
                        chunks.append(sub)
                    sub = s
                else:
                    sub = s if not sub else sub + s
            if len(sub) >= min_len:
                chunks.append(sub)
            continue
        if len(cur) + len(p) <= max_len:
            cur = p if not cur else cur + '\n' + p
        else:
            flush()
            cur = p
    flush()

    # 改进1: 兜底硬切 >hard_max 字符的超长块
    final = []
    for c in chunks:
        if len(c) > hard_max:
            final.extend(_hard_split(c, max_len))
        else:
            final.append(c)
    return final


# ---------------------------------------------------------------------------
# MinHash 去重
# ---------------------------------------------------------------------------
def minhash_of(text, num_perm=128):
    """字符级 MinHash (jieba 对超长/特殊文本不稳定, 去重用字符 n-gram 足够)."""
    if len(text) > 5000:
        text = text[:5000]
    # 用 2-gram 字符 token; 用普通循环避免 set comprehension 在长文本上的闭包异常
    tokens = set()
    tlen = len(text)
    for i in range(tlen - 1):
        tokens.add(text[i:i + 2])
    mh = MinHash(num_perm=num_perm)
    for t in tokens:
        mh.update(t.encode('utf-8'))
    return mh


def dedup_minhash(texts, threshold=0.8, num_perm=128, batch_size=5000):
    """MinHashLSH 去重 (流式批处理, 控制内存). 返回保留索引列表."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, t in enumerate(texts):
        mh = minhash_of(t, num_perm)
        dup = lsh.query(mh)
        if not dup:
            lsh.insert(i, mh)
            keep.append(i)
        if (i + 1) % batch_size == 0:
            print(f'  [dedup] {i+1}/{len(texts)} (kept {len(keep)})', flush=True)
    return keep


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def collect_sources():
    """收集三源原始文本."""
    sources = []  # (name, text)
    # 1. corpus.txt — 大文件, 按行块读取, 避免单超长文本
    corpus = os.path.join(PROJECT_ROOT, '..', 'esp32-ai', 'data_v4', 'corpus.txt')
    if os.path.exists(corpus):
        with open(corpus, 'r', encoding='utf-8', errors='replace') as f:
            buf, block = [], []
            for line in f:
                block.append(line)
                if len('\n'.join(block)) >= 100000:  # 100K 字符/块
                    buf.append(''.join(block))
                    block = []
            if block:
                buf.append(''.join(block))
        for i, b in enumerate(buf):
            sources.append((f'corpus.txt#part{i}', b))
        print(f'[source] corpus.txt: {os.path.getsize(corpus)/1e6:.1f} MB -> {len(buf)} parts')
    # 2. medica + 临床诊疗指南全集 md (兼容 Windows D:\ 与 WSL /mnt/d/ 路径)
    md_bases = []
    for base in ['D:/docs/raw/medica', 'D:/docs/raw/临床诊疗指南全集']:
        if os.path.isdir(base):
            md_bases.append(base)
        else:
            alt = '/mnt/d/docs/raw/' + os.path.basename(base)
            if os.path.isdir(alt):
                md_bases.append(alt)
    for base in md_bases:
        for md in glob.glob(os.path.join(base, '**', '*.pdf.md'), recursive=True):
            name = os.path.relpath(md, base)
            with open(md, 'r', encoding='utf-8', errors='replace') as f:
                sources.append((name, f.read()))
    print(f'[source] total {len(sources)} raw docs')
    return sources


def main():
    ap = argparse.ArgumentParser(description='管线A: 医疗 Pretrain 语料构建')
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'dataset', 'pretrain_medical.jsonl'))
    ap.add_argument('--report', default=os.path.join(PROJECT_ROOT, 'out', 'medical_pretrain_report.json'))
    ap.add_argument('--min_len', type=int, default=80)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--threshold', type=float, default=0.8, help='MinHash 去重 Jaccard 阈值')
    ap.add_argument('--num_perm', type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    sources = collect_sources()

    # 清洗 + 分块
    all_chunks = []
    cleaned_chars = 0
    raw_chars = 0
    for name, text in sources:
        raw_chars += len(text)
        clean = clean_markdown(text)
        cleaned_chars += len(clean)
        for c in chunk_text(clean, args.min_len, args.max_len):
            all_chunks.append({'text': c, 'src': name})

    print(f'[clean] {raw_chars/1e6:.1f}M chars -> {cleaned_chars/1e6:.1f}M chars '
          f'({100*cleaned_chars/max(raw_chars,1):.1f}% kept)')
    print(f'[chunk] {len(all_chunks)} chunks')

    # MinHash 去重
    texts = [c['text'] for c in all_chunks]
    keep_idx = dedup_minhash(texts, args.threshold, args.num_perm)
    print(f'[dedup] {len(texts)} -> {len(keep_idx)} kept (threshold={args.threshold})')

    # 写输出
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for i in keep_idx:
            f.write(json.dumps({'text': texts[i]}, ensure_ascii=False) + '\n')
            n_written += 1
    print(f'[out] {args.out}: {n_written} samples, {os.path.getsize(args.out)/1e6:.1f} MB')

    # 质量报告
    lens = [len(texts[i]) for i in keep_idx]
    report = {
        'raw_docs': len(sources),
        'raw_chars': raw_chars,
        'cleaned_chars': cleaned_chars,
        'chunks_before_dedup': len(texts),
        'chunks_after_dedup': n_written,
        'dedup_rate': 1 - n_written / max(len(texts), 1),
        'threshold': args.threshold,
        'avg_len': sum(lens) / max(len(lens), 1),
        'min_len': min(lens) if lens else 0,
        'max_len': max(lens) if lens else 0,
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[report] {args.report}')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
