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

sft = load('full_sft_h2')
dpo = load('dpo_h2')

questions = [
    '介绍一下你自己',
    '如何缓解工作压力？',
    '什么是机器学习？',
    '写一首关于秋天的诗',
]

def gen(model, q, max_new=120):
    messages = [{'role': 'user', 'content': q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.85, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    dt = time.time() - t0
    gen = out[0][ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), dt, gen.shape[0]

for q in questions:
    print('\n[Q] {}'.format(q))
    a1, t1, n1 = gen(sft, q)
    a2, t2, n2 = gen(dpo, q)
    print('[SFT] {}...'.format(a1[:150]))
    print('      ({:.1f}s, {:.0f} tok/s)'.format(t1, n1/max(t1,1e-6)))
    print('[DPO] {}...'.format(a2[:150]))
    print('      ({:.1f}s, {:.0f} tok/s)'.format(t2, n2/max(t2,1e-6)))
PYEOF
