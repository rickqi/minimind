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
model = MiniMindForCausalLM(cfg)
sd = torch.load('out/full_sft_h2_384_ple.pth', map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=True)
print('load strict OK: missing={} unexpected={}'.format(len(missing), len(unexpected)))
b = model.param_budget()
print('H2 params: core {:.2f}M + table {:.2f}M + stream {:.2f}M = {:.2f}M'.format(
    b['core']/1e6, b['table']/1e6, b['stream']/1e6, b['total']/1e6))
print('H2 sizes: fp32 {:.1f}MB | fp16 {:.1f}MB | int4 {:.1f}MB'.format(
    b['total']*4/1e6, b['total']*2/1e6, b['total']*0.5/1e6))
model = model.cuda().eval()
tok = AutoTokenizer.from_pretrained('model')

questions = [
    '介绍一下你自己',
    '什么是机器学习？',
    '推荐一些杭州的特色美食吧',
    '如何缓解工作压力？',
    '请解释什么是光合作用',
    '写一篇关于秋天的短文',
]

for q in questions:
    messages = [{'role': 'user', 'content': q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, temperature=0.85, top_p=0.85,
                             top_k=50, do_sample=True, eos_token_id=2)
    dt = time.time() - t0
    gen = out[0][ids.shape[1]:]
    n_new = gen.shape[0]
    text = tok.decode(gen, skip_special_tokens=True)
    print('\n[Q] {}'.format(q))
    print('[A] {}'.format(text[:180]))
    print('    ({:.1f}s, {:.1f} tok/s)'.format(dt, n_new/max(dt,1e-6)))
PYEOF
