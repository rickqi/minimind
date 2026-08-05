#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch, time
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

cfg = MiniMindConfig(hidden_size=384, num_hidden_layers=8, use_ple=True, ple_dim=128)
tok = AutoTokenizer.from_pretrained('model')

def load(weight):
    model = MiniMindForCausalLM(cfg)
    sd = torch.load('out/{}_384_ple.pth'.format(weight), map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.cuda().eval()

models = {
    '原H2': load('full_sft_h2'),
    '混合增强': load('full_sft_h2_med'),
    '纯医学': load('full_sft_h2_pure'),
}

questions = [
    '高血压的诊断标准是什么？',
    '糖尿病的临床表现有哪些？',
    '病毒性肝炎的治疗原则是什么？',
    '肺癌的早期症状有哪些？',
    '感染性休克的血象检查有什么特点？',
    '介绍一下你自己',
]

def gen(model, q, max_new=120):
    messages = [{'role': 'user', 'content': q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.7, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    gen = out[0][ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)

for q in questions:
    print('\n[Q] {}'.format(q))
    for name, m in models.items():
        a = gen(m, q)
        print('[{}] {}'.format(name, a[:130]))
PYEOF
