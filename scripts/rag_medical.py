#!/usr/bin/env python3
"""
PC 端 RAG 方案 (移植 esp32-ai 蓝图, 适配 MiniMind H1/H2)

核心: IDF 加权倒排索引 + Top-2 证据注入 + ChatML 模板 (esp32-ai 已验证有效)

组件:
  1. build_index: format_data.jsonl (11K 医学 QA) -> 倒排索引 + IDF
  2. retrieve:   IDF 加权评分, Top-2 证据 (esp32-ai: Top-1 会退化, Top-2 提供冗余)
  3. 注入:       ChatML system+user 消息携带证据 (尊重 minimind 训练分布)

用法:
  python scripts/rag_medical.py build            # 构建索引 (缓存到 out/rag_index.pkl)
  python scripts/rag_medical.py query "肺癌早期症状"   # 检索测试
  python scripts/rag_medical.py chat "肺癌早期症状"    # 检索+注入+生成 (H2)
"""

import argparse
import json
import math
import os
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jieba

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(PROJECT_ROOT, '..', 'esp32-ai', 'data_v4', 'kb', 'format_data.jsonl')
INDEX_PATH = os.path.join(PROJECT_ROOT, 'out', 'rag_index.pkl')
DOC_CHARS = 60  # 证据截断长度 (esp32-ai P2 发现: 仅答案, 短截断)
IDF_SCALE = 64.0


def terms_of(text):
    """jieba 分词 + 单字补充 (提升检索精准度)."""
    toks = [t for t in jieba.cut(text) if t.strip() and len(t.strip()) > 0]
    # 去停用词 (单字虚词/标点)
    stop = set('的了是在和有就不都而及与或一个中其') | set('，。、；：！？""''（）【】\n \t')
    toks = [t for t in toks if t not in stop and not t.isspace()]
    return set(toks)


def build_index(entries):
    """jieba 分词倒排索引 + IDF (esp32-ai 建议的检索升级)."""
    docs = [(a[:DOC_CHARS], label) for q, a, label in entries]
    inverted = {}
    for di, (q, a, label) in enumerate(entries):
        q = q or ""
        a = a or ""
        for term in terms_of(q + a[:DOC_CHARS]):
            inverted.setdefault(term, []).append(di)
    N = len(docs)
    idf = {}
    for t, dlist in inverted.items():
        df = len(dlist)
        idf[t] = min(int(IDF_SCALE * math.log(1.0 + N / max(df, 1))), 255)
    return docs, inverted, idf


def retrieve(query, doclists, docs, idf, k=3):
    """IDF 加权倒排检索 (jieba 分词)."""
    q_terms = terms_of(query)
    sc = {}
    for term in q_terms:
        if term not in idf:
            continue
        w = idf[term]
        for d in doclists.get(term, ()):
            sc[d] = sc.get(d, 0) + w
    top = sorted(sc, key=sc.get, reverse=True)[:k]
    return top, sc


def load_kb():
    entries = []
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            entries.append((d.get('question', ''), d.get('answer', ''), d.get('label', '')))
    return entries


def cmd_build():
    print(f'加载 KB: {KB_PATH}')
    entries = load_kb()
    docs, inverted, idf = build_index(entries)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump({'docs': docs, 'inverted': inverted, 'idf': idf}, f)
    print(f'索引构建完成: {len(docs)} docs, {len(idf)} terms -> {INDEX_PATH}')
    # 验证 12 个 esp32-ai 测试问题
    tests = ["感冒发烧", "肺癌早期症状", "急性重型肝炎", "糖尿病酮症酸中毒",
             "白疕皮损特点", "宫外孕如何治疗", "高血压用药", "带状疱疹后遗神经痛",
             "肝硬化腹水治疗", "儿童肺炎支原体感染", "肝豆状核变性", "心肌梗死急救"]
    print(f'\n{"问题":<14} {"top1命中":<30}')
    for q in tests:
        top, scores = retrieve(q, inverted, docs, idf)
        if top:
            doc, label = docs[top[0]]
            print(f'{q:<14} ({label}) {doc[:26]}')
        else:
            print(f'{q:<14} 无匹配')


def cmd_query(query):
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    docs, inverted, idf = data['docs'], data['inverted'], data['idf']
    top, scores = retrieve(query, inverted, docs, idf, k=2)
    print(f'检索 "{query}" -> Top-{len(top)} 证据:')
    for i, d in enumerate(top):
        doc, label = docs[d]
        print(f'  [{i+1}] ({label}) {doc}')


def cmd_chat(query, hidden_size=384, n_layers=8, ple_dim=128, weight='full_sft_h2'):
    """检索 + 证据注入 + 生成 (ChatML 模板)."""
    import torch
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer

    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    docs, inverted, idf = data['docs'], data['inverted'], data['idf']
    top, scores = retrieve(query, inverted, docs, idf, k=2)
    evidence = '\n'.join(docs[d][0] for d in top) if top else ''

    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=n_layers,
                         use_ple=True, ple_dim=ple_dim)
    model = MiniMindForCausalLM(cfg)
    sd = torch.load(f'out/{weight}_{hidden_size}_ple.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained('model')

    # ChatML 证据注入 (尊重训练分布)
    if evidence:
        messages = [
            {'role': 'system', 'content': '你是一个医学助手，请根据提供的参考资料准确回答问题。'},
            {'role': 'user', 'content': f'参考资料：\n{evidence}\n\n问题：{query}'},
        ]
    else:
        messages = [{'role': 'user', 'content': query}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    print(f'[RAG] 注入 {len(top)} 条证据')
    if top:
        print(f'[证据] {evidence[:80]}...')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, temperature=0.7, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    gen = out[0][ids.shape[1]:]
    print(f'\n[A] {tok.decode(gen, skip_special_tokens=True)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MiniMind H1/H2 医学 RAG')
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('build')
    p_q = sub.add_parser('query')
    p_q.add_argument('query')
    p_c = sub.add_parser('chat')
    p_c.add_argument('query')
    p_c.add_argument('--hidden_size', type=int, default=384)
    p_c.add_argument('--n_layers', type=int, default=8)
    p_c.add_argument('--ple_dim', type=int, default=128)
    p_c.add_argument('--weight', default='full_sft_h2')
    args = ap.parse_args()
    if args.cmd == 'build':
        cmd_build()
    elif args.cmd == 'query':
        cmd_query(args.query)
    elif args.cmd == 'chat':
        cmd_chat(args.query, args.hidden_size, args.n_layers, args.ple_dim, args.weight)
    else:
        ap.print_help()
