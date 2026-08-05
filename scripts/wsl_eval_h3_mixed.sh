#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

def load(hidden, layers, ple, weight):
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_ple=True, ple_dim=ple)
    m = MiniMindForCausalLM(cfg)
    m.load_state_dict(torch.load(f'out/{weight}_{hidden}_ple.pth', map_location='cpu', weights_only=True), strict=True)
    return m.cuda().eval()

tok = AutoTokenizer.from_pretrained('model')

# 对比: 原H3(纯通用从零) vs H3混合(医疗1:2从零+混合SFT) vs H2 RAG+RAFT(最优)
models = {
    'H3原(通用)': load(512, 8, 128, 'full_sft_h3'),
    'H3混合(新)': load(512, 8, 128, 'full_sft_h3_mixed'),
}

questions = [
    '肺癌的早期症状有哪些',
    '高血压的诊断标准是什么',
    '糖尿病的临床表现有哪些',
    '病毒性肝炎的治疗原则是什么',
    '介绍一下你自己',
]

def gen(model, q, max_new=120):
    messages = [{'role': 'user', 'content': q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.7, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

for q in questions:
    print('\n[Q] {}'.format(q))
    for name, m in models.items():
        a = gen(m, q)
        print('[{}] {}'.format(name, a[:130]))
PYEOF
