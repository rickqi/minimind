#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch, time
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=6, use_ple=True, ple_dim=96)
model = MiniMindForCausalLM(cfg).cuda()
b = model.param_budget()
print('PLE params: core {:.2f}M + table {:.2f}M + stream {:.2f}M = {:.2f}M'.format(
    b['core']/1e6, b['table']/1e6, b['stream']/1e6, b['total']/1e6))
x = torch.randint(0, 6400, (8, 128)).cuda()
y = torch.randint(0, 6400, (8, 128)).cuda()
from torch import optim
opt = optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(3):
    out = model(x, labels=y); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
t0 = time.time()
n = 20
for _ in range(n):
    out = model(x, labels=y); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
dt = (time.time()-t0)/n
print('GPU TRAIN: {:.3f}s/step (bs8,seq128) -> {:.0f} step/min'.format(dt, 60/dt))
rows = 1270238
print('full pretrain 1 epoch bs16: {:.0f} steps -> {:.1f} min'.format(rows/16, (rows/16)*dt/60))
print('20K steps -> {:.1f} min'.format(20000*dt/60))
"
