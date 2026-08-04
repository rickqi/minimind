#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

# 验证 DPO int4 权重可加载 (quantize_ple.py 产物结构)
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=6, use_ple=True, ple_dim=96)
tok = AutoTokenizer.from_pretrained('model')

sd = torch.load('models/dpo_h1_256_int4_g32.pth', map_location='cpu', weights_only=True)
print('dpo_h1 int4 keys: {} (dict tensors: {})'.format(
    len(sd), sum(1 for v in sd.values() if isinstance(v, dict))))
codes0 = sd['model.embed_tokens.weight']['codes']
print('embed codes shape: {}, dtype: {}'.format(tuple(codes0.shape), codes0.dtype))
print('embed scales shape: {}'.format(tuple(sd['model.embed_tokens.weight']['scales'].shape)))
print('norm kept fp16: {}'.format(sd['model.norm.weight'].dtype))
print('OK: DPO int4 weights loadable')
PYEOF
