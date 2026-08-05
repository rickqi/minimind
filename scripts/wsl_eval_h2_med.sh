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

base = load('full_sft_h2')
med = load('full_sft_h2_med')

questions = [
    '什么是高血压？诊断标准是什么？',
    '糖尿病的临床表现有哪些？',
    '病毒性肝炎如何治疗？',
    '肺癌的早期症状有哪些？',
    '感冒与流感的区别是什么？',
    '如何缓解头痛？',
]

def gen(model, q, max_new=150):
    messages = [{'role': 'user', 'content': q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.7, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    dt = time.time() - t0
    gen = out[0][ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), dt, gen.shape[0]

for q in questions:
    print('\n[Q] {}'.format(q))
    a1, _, _ = gen(base, q)
    a2, _, _ = gen(med, q)
    print('[原H2] {}'.format(a1[:160]))
    print('[医疗] {}'.format(a2[:160]))
PYEOF
