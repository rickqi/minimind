#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys, os, math, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scripts')

import jieba
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# ========== RAG 检索 (jieba IDF) ==========
KB_PATH = '/mnt/d/codes/esp32-ai/data_v4/kb/format_data.jsonl'
DOC_CHARS = 60
STOP = set('的了是在和有就不都而及与或一个中其') | set('，。、；：！？""\'\'（）【】\n \t')

def terms_of(text):
    return set(t for t in jieba.cut(text) if t.strip() and t not in STOP and not t.isspace())

def load_kb():
    entries = []
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            entries.append((d.get('question',''), d.get('answer',''), d.get('label','')))
    return entries

def build_index(entries):
    docs = [(a[:DOC_CHARS], label) for q,a,label in entries]
    inv = {}
    for di,(q,a,label) in enumerate(entries):
        for term in terms_of((q or '')+(a or '')[:DOC_CHARS]):
            inv.setdefault(term, []).append(di)
    N = len(docs); idf = {}
    for t,dl in inv.items():
        idf[t] = min(int(64.0*math.log(1.0+N/max(len(dl),1))), 255)
    return docs, inv, idf

def retrieve(query, inv, idf, k=2):
    sc = {}
    for term in terms_of(query):
        if term in idf:
            for did in inv.get(term, ()):
                sc[did] = sc.get(did,0) + idf[term]
    return sorted(sc, key=sc.get, reverse=True)[:k]

entries = load_kb()
docs, inv, idf = build_index(entries)
print(f'KB: {len(docs)} docs, {len(idf)} terms')

# ========== 模型加载 ==========
def load(hidden, layers, ple, weight):
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_ple=True, ple_dim=ple)
    m = MiniMindForCausalLM(cfg)
    m.load_state_dict(torch.load(f'out/{weight}_{hidden}_ple.pth', map_location='cpu', weights_only=True), strict=True)
    return m.cuda().eval()

tok = AutoTokenizer.from_pretrained('model')

models = {
    'H1 RAFT (10.8M)': load(256, 6, 96, 'full_sft_h1_raft'),
    'H2 RAFT (25.0M)': load(384, 8, 128, 'full_sft_h2_raft_v3'),
    'H3混合 (38.2M)': load(512, 8, 128, 'full_sft_h3_mixed'),
    'H3混合+RAFT (38.2M)': load(512, 8, 128, 'full_sft_h3_mixed_raft'),
}

def gen(model, messages, max_new=100):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.7, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def make_messages(q, evidence):
    if evidence:
        return [
            {'role':'system','content':'你是一个医学助手，请根据提供的参考资料准确回答问题。'},
            {'role':'user','content':f'参考资料：\n{evidence}\n\n问题：{q}'},
        ]
    return [{'role':'user','content':q}]

questions = [
    '肺癌的早期症状有哪些',
    '高血压的诊断标准是什么',
    '糖尿病的临床表现有哪些',
    '病毒性肝炎的治疗原则是什么',
    '介绍一下你自己',
]

for q in questions:
    print('\n' + '='*72)
    print('Q:', q)
    top = retrieve(q, inv, idf, k=2)
    ev = '\n'.join(docs[d][0] for d in top)
    print('[RAG证据]', ev[:60].replace('\n',' | '))
    for name, m in models.items():
        use_rag = 'RAFT' in name  # RAFT 模型走 RAG, H3混合走内在知识
        a = gen(m, make_messages(q, ev if use_rag else None))
        print(f'  [{name}] {a[:110]}')
PYEOF
