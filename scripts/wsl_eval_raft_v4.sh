#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys, os, math, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scripts')

import jieba
jieba.load_userdict('out/medical_jieba.txt')  # 医学词典
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

KB_PATH = '/mnt/d/codes/esp32-ai/data_v4/kb/format_data.jsonl'
DOC_CHARS = 60
STOP = set('的了是在和有就不都而及与或一个中其') | set('，。、；：！？""\'\'（）【】\n \t')
NON_MED = ['健康管理', '理赔', '产品条款', '销售', '消保']

def is_med(lb):
    return not any(k in lb for k in NON_MED)

def terms_of(t):
    return set(x for x in jieba.cut(t) if x.strip() and x not in STOP and not x.isspace())

entries = []
with open(KB_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        entries.append((d.get('question',''), d.get('answer',''), d.get('label','')))
# med_only 过滤
entries = [e for e in entries if is_med(e[2])]
docs = [(a[:DOC_CHARS], lb) for q,a,lb in entries]
inv = {}
for di,(q,a,lb) in enumerate(entries):
    for term in terms_of((q or '')+(a or '')[:DOC_CHARS]):
        inv.setdefault(term,[]).append(di)
N=len(docs); idf={}
for t,dl in inv.items():
    idf[t]=min(int(64.0*math.log(1.0+N/max(len(dl),1))),255)

def retrieve(q,k=2):
    sc={}
    for term in terms_of(q):
        if term in idf:
            for did in inv.get(term,()):
                sc[did]=sc.get(did,0)+idf[term]
    return sorted(sc,key=sc.get,reverse=True)[:k]

tok = AutoTokenizer.from_pretrained('model')
cfg = MiniMindConfig(hidden_size=384, num_hidden_layers=8, use_ple=True, ple_dim=128)
m = MiniMindForCausalLM(cfg)
m.load_state_dict(torch.load('out/full_sft_h2_raft_v4_384_ple.pth', map_location='cpu', weights_only=True), strict=True)
m = m.cuda().eval()

def gen(messages, max_new=80):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new, temperature=0.7, top_p=0.85,
                         top_k=50, do_sample=True, eos_token_id=2)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def rag(q):
    top = retrieve(q)
    ev = '\n'.join(docs[d][0] for d in top)
    return [{'role':'system','content':'你是一个医学助手，请根据提供的参考资料准确回答问题。'},
            {'role':'user','content':f'参考资料：\n{ev}\n\n问题：{q}'}]

def no_rag(q):
    return [{'role':'user','content':q}]

questions = ['肺癌的早期症状有哪些', '高血压的诊断标准是什么', '介绍一下你自己']
for q in questions:
    print('\n' + '='*60)
    print('Q:', q)
    if q == '介绍一下你自己':
        a = gen(no_rag(q))
        print('[无RAG(修后)]', a[:80])
    else:
        top = retrieve(q)
        ev = '\n'.join(docs[d][0] for d in top)
        print('[证据]', ev[:50].replace('\n',' | '))
        a = gen(rag(q))
        print('[RAG(新v4)]', a[:100])
PYEOF
