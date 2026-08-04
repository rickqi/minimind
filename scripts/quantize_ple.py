"""
PLE 模型 int4 量化导出 (移植 esp32-ai src/quantize.py 思路)

对训练好的 PLE 模型 (H1/H2) 做 group-wise symmetric int4 PTQ:
  - group=32 (SFT 模型对 4-bit 敏感, esp32-ai 实测 group=128 会崩, group=32 可用)
  - 所有 >=2D 权重量化 (含 PLE table), norms 保持 fp32
  - 输出: 量化后权重 (可直接部署的 int4 表示) + 量化前后 val loss 退化报告

用法:
    python scripts/quantize_ple.py --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 \
        --weight full_sft_h1 --group 32
"""

import argparse
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from torch.utils.data import DataLoader


def quantize_groupwise(w, bits=4, group=32, fp16_scales=True):
    """Symmetric group-wise quant along last dim; returns dequantized fp32.
    移植自 esp32-ai src/quantize.py (GGUF-Q4 style).
    """
    orig_shape = w.shape
    x = w.reshape(-1, orig_shape[-1]).float()
    out, cols = x.shape
    pad = (group - cols % group) % group
    if pad:
        x = torch.cat([x, torch.zeros(out, pad, device=x.device)], dim=1)
    x = x.reshape(out, -1, group)
    qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
    scale = x.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    if fp16_scales:
        scale = scale.half().float()
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    dq = (q * scale).reshape(out, -1)[:, :cols]
    return dq.reshape(orig_shape).to(w.dtype)


def quantize_model(model, bits=4, group=32, quant_table=True, fp16_scales=True):
    """原地量化模型全部 >=2D 权重 (含 PLE table), norms 保持 fp32."""
    n_q, n_skip = 0, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2:
                n_skip += 1
                continue  # norms -> fp32
            if 'table' in name and not quant_table:
                n_skip += 1
                continue
            if torch.isnan(p).any() or torch.isinf(p).any():
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
            p.copy_(quantize_groupwise(p.data, bits, group, fp16_scales))
            n_q += p.numel()
    return n_q, n_skip


@torch.no_grad()
def val_loss(model, tokenizer, data_path, max_seq_len, device, iters=100, seed=1234):
    """在训练数据子集上测量 loss (用 PretrainDataset 兼容格式)."""
    ds = PretrainDataset(data_path, tokenizer, max_length=max_seq_len)
    # 随机采样固定步数
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(iters, len(ds)), replace=False)
    model.eval()
    losses = []
    for i in idxs:
        x, y = ds[i]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        out = model(x, labels=y)
        losses.append(out.loss.item())
    return sum(losses) / len(losses)


def export_int4_weights(model, save_path):
    """导出 int4 权重: codes(int8 打包) + fp16 scales, 便于部署侧重建."""
    state = {}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            state[name] = p.half().cpu()  # norms fp16
            continue
        # 重新计算 codes + scales
        x = p.float().reshape(-1, p.shape[-1])
        out, cols = x.shape
        group = 32
        pad = (group - cols % group) % group
        if pad:
            x = torch.cat([x, torch.zeros(out, pad)], dim=1)
        xg = x.reshape(out, -1, group)
        qmax = 7
        scale = (xg.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8).half()
        q = torch.clamp(torch.round(xg / scale.float()), -qmax, qmax).to(torch.int8)
        state[name] = {'codes': q.cpu(), 'scales': scale.squeeze(-1).cpu()}
    torch.save(state, save_path)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--num_hidden_layers', type=int, default=6)
    ap.add_argument('--ple_dim', type=int, default=96)
    ap.add_argument('--weight', type=str, default='full_sft_h1', help='权重前缀 (full_sft_h1 / full_sft_h2)')
    ap.add_argument('--save_dir', type=str, default='out', help='训练权重目录 (fp16 .pth 输入)')
    ap.add_argument('--export_dir', type=str, default='models', help='int4 部署产物导出目录 (默认 models/, 与训练产物 out/ 分离)')
    ap.add_argument('--group', type=int, default=32)
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--data_path', type=str, default='dataset/pretrain_t2t_mini.jsonl')
    ap.add_argument('--max_seq_len', type=int, default=128)
    ap.add_argument('--val_iters', type=int, default=100)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                         use_ple=True, ple_dim=args.ple_dim)
    suffix = '_ple'
    ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{suffix}.pth'
    print(f'loading: {ckp}')
    model = MiniMindForCausalLM(cfg)
    sd = torch.load(ckp, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    b = model.param_budget()
    print('params: core {:.2f}M + table {:.2f}M + stream {:.2f}M = {:.2f}M'.format(
        b['core']/1e6, b['table']/1e6, b['stream']/1e6, b['total']/1e6))
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained('model')

    print(f'\n=== int{args.bits} group={args.group} PTQ 评估 ===')
    fp = val_loss(model, tokenizer, args.data_path, args.max_seq_len, args.device, args.val_iters)
    print(f'fp32  val {fp:.4f} (ppl {math.exp(fp):.2f})')

    # 量化
    qmodel = MiniMindForCausalLM(cfg)
    qmodel.load_state_dict(torch.load(ckp, map_location='cpu', weights_only=True), strict=True)
    qmodel = qmodel.to(args.device).eval()
    nq, ns = quantize_model(qmodel, args.bits, args.group, quant_table=True, fp16_scales=True)
    q = val_loss(qmodel, tokenizer, args.data_path, args.max_seq_len, args.device, args.val_iters)
    print(f'int{args.bits} val {q:.4f} (ppl {math.exp(q):.2f}) | deg {q-fp:+.4f} | quantized {nq/1e6:.2f}M params | norms kept {ns}')

    # 导出 int4 权重 (部署产物 -> 独立 export_dir, 与训练产物 out/ 分离)
    os.makedirs(args.export_dir, exist_ok=True)
    export_path = f'{args.export_dir}/{args.weight}_{args.hidden_size}_int{args.bits}_g{args.group}.pth'
    state = export_int4_weights(qmodel, export_path)
    # 计算导出文件大小
    size = sum(
        (v['codes'].nbytes + v['scales'].nbytes) if isinstance(v, dict) else v.nbytes
        for v in state.values()
    )
    print(f'\nint4 导出: {export_path}')
    print(f'导出尺寸: {size/1e6:.1f} MB (codes+scales, 可进一步打包为二进制)')
    print(f'理论 int4 权重: {b["total"]*0.5/1e6:.1f} MB')


if __name__ == '__main__':
    main()
