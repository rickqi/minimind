"""
Export a trained MiniMind PLE model to PLE1 flat binary (esp32-ai format).

对齐 esp32-ai src/export.py 的 PLE1 格式, 供 C 运行时 mmap + 烧录:
  [header: magic + int32 config fields + float rope_theta]
  then, in fixed order, each tensor as either
    QUANT: int4 codes packed 2-per-byte (group-wise, group=G along last dim)
           followed by fp16 scales, one per group
    FP32 : raw fp32 (norms only -- tiny)

Quantization matches scripts/quantize_ple.py (symmetric group-wise int4),
so golden logits are the *4-bit* model's logits.

用法:
    python scripts/export_ple1.py --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 \
        --weight full_sft_h1 --out_dir models
"""

import argparse
import os
import struct
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

MAGIC = 0x504C4531  # "PLE1"
GROUP = 32  # SFT 模型 4-bit 敏感, group=32 (esp32-ai 实测 group=128 崩)


def quant_pack(w, group=GROUP, bits=4):
    """Group-wise symmetric int4/int8, ragged (no padding) with fp16 scales.
    与 esp32-ai src/export.py:34 完全一致: 末组可短于 group, scales 做 fp16 舍入.
    bits=4: codes 2-per-byte; bits=8: codes 1-per-byte (int8, -127..127).
    """
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    max_code = 7 if bits == 4 else 127
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / max_code).clamp_min(1e-8)
        sc = sc.half().float()  # fp16-round the scale
        scales[:, gi] = sc.squeeze(1)
        qi = torch.clamp(torch.round(seg / sc), -max_code, max_code)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)
    scales16 = scales.numpy().astype(np.float16)
    if bits == 8:
        codes = q.to(torch.int8).numpy()          # -127..127, 1 byte per code
        return codes.reshape(-1), scales16.reshape(-1), dq
    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()  # rows x cols, 0..15
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    return packed.reshape(-1), scales16.reshape(-1), dq


def main():
    ap = argparse.ArgumentParser(description='Export MiniMind PLE model to PLE1 flat binary')
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--num_hidden_layers', type=int, default=6)
    ap.add_argument('--num_attention_heads', type=int, default=8, help='q_heads (默认与 minimind 一致=8)')
    ap.add_argument('--num_key_value_heads', type=int, default=4, help='kv_heads (GQA)')
    ap.add_argument('--ple_dim', type=int, default=96)
    ap.add_argument('--seq_len', type=int, default=128, help='训练 max_seq_len, 写入 header 供 C 端 KV 上限')
    ap.add_argument('--weight', type=str, default='full_sft_h1')
    ap.add_argument('--save_dir', type=str, default='out', help='训练权重目录')
    ap.add_argument('--out_dir', type=str, default='models')
    ap.add_argument('--group', type=int, default=GROUP)
    ap.add_argument('--bits', type=int, default=4, choices=[4, 8], help='quantization bits (4=default, 8=low-loss)')
    args = ap.parse_args()

    cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_ple=True, ple_dim=args.ple_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )
    ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}_ple.pth'
    print(f'loading: {ckp}')
    model = MiniMindForCausalLM(cfg)
    model.load_state_dict(torch.load(ckp, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    b = model.param_budget()
    print('params: core {:.2f}M + table {:.2f}M + stream {:.2f}M = {:.2f}M'.format(
        b['core']/1e6, b['table']/1e6, b['stream']/1e6, b['total']/1e6))

    sd = model.state_dict()

    # GQA -> MHA 转换: esp32-ai llm.h 是标准 MHA (q/k/v 同用 n_heads 头),
    # MiniMind 是 GQA (q_heads=8, kv_heads=4)。将 k_proj/v_proj 的 kv 头
    # repeat_interleave 复制为 q_heads 头, 数学等价 (MHA 每 2 个 q 头共享 kv)。
    # [kv_heads*head_dim, D] -> [q_heads*head_dim, D] = [D, D]
    if cfg.num_key_value_heads < cfg.num_attention_heads:
        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        hd = cfg.head_dim
        for i in range(cfg.num_hidden_layers):
            for proj in ('k_proj', 'v_proj'):
                key = f'model.layers.{i}.self_attn.{proj}.weight'
                w = sd[key]
                assert w.shape[0] == cfg.num_key_value_heads * hd, \
                    f'{key} shape {w.shape} != kv_heads*head_dim'
                w2 = w.view(cfg.num_key_value_heads, hd, -1).repeat_interleave(n_rep, dim=0).reshape(-1, w.shape[1])
                sd[key] = w2
        print(f'GQA->MHA: kv_heads {cfg.num_key_value_heads} -> {cfg.num_attention_heads} (n_rep={n_rep})')

    # 固定顺序的 tensor 计划 (C 端按此顺序硬编码读取)
    # minimind 命名 -> 语义 (esp32-ai 命名)
    #   embed_tokens.weight (tied=lm_head) -> tok_emb.weight
    #   ple_table.weight                   -> ple_table.weight
    #   ple_model_proj.weight              -> ple_model_proj.weight
    #   ple_proj_norm.weight               -> ple_proj_norm.weight
    #   layers.{i}.self_attn.{q,k,v,o}_proj -> attn.qkv/proj (分开存, 语义同)
    #   layers.{i}.input_layernorm         -> attn_norm
    #   layers.{i}.post_attention_layernorm-> ffn_norm
    #   layers.{i}.mlp.{gate,up,down}_proj -> ffn.gate/up/down
    #   layers.{i}.ple_{gate,proj,norm}    -> ple_gate/ple_proj/ple_norm
    #   norm.weight                        -> out_norm
    plan = []
    def add(key, quant):
        plan.append((key, sd[key], quant))

    add('model.embed_tokens.weight', True)   # tied: input embed + output head
    add('model.ple_model_proj.weight', True)
    add('model.ple_proj_norm.weight', False)
    add('model.ple_table.weight', True)      # 稀疏每层嵌入表
    for i in range(cfg.num_hidden_layers):
        p = f'model.layers.{i}.'
        add(p + 'self_attn.q_proj.weight', True)
        add(p + 'self_attn.k_proj.weight', True)
        add(p + 'self_attn.v_proj.weight', True)
        add(p + 'self_attn.o_proj.weight', True)
        add(p + 'self_attn.q_norm.weight', False)
        add(p + 'self_attn.k_norm.weight', False)
        add(p + 'input_layernorm.weight', False)
        add(p + 'post_attention_layernorm.weight', False)
        add(p + 'mlp.gate_proj.weight', True)
        add(p + 'mlp.up_proj.weight', True)
        add(p + 'mlp.down_proj.weight', True)
        add(p + 'ple_gate.weight', True)
        add(p + 'ple_proj.weight', True)
        add(p + 'ple_norm.weight', False)
    add('model.norm.weight', False)

    # 重建反量化 state dict 驱动 golden forward
    dq_sd = {k: v.clone() for k, v in sd.items()}
    blobs = []
    for name, t, quant in plan:
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f'  cleanup NaN/Inf in {name}: {torch.isnan(t).sum().item()} NaN')
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if quant:
            packed, scales, dq = quant_pack(t, args.group, args.bits)
            dq_sd[name] = dq
            blobs.append(('Q', name, t.shape, packed, scales))
        else:
            blobs.append(('F', name, t.shape, t.contiguous().numpy().astype(np.float32), None))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'{args.weight}_h{args.hidden_size}_ple1.bin')
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        for v in [cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers,
                  cfg.num_attention_heads, cfg.intermediate_size, cfg.ple_dim,
                  args.seq_len, args.group]:
            f.write(struct.pack('<i', v))
        f.write(struct.pack('<f', cfg.rope_theta))
        for entry in blobs:
            kind = entry[0]
            if kind == 'Q':
                _, _, _, packed, scales = entry
                f.write(struct.pack('<i', args.group))  # per-tensor group
                f.write(packed.tobytes())
                f.write(scales.tobytes())
            else:
                _, _, _, arr, _ = entry
                f.write(arr.tobytes())
    size = os.path.getsize(path)
    print(f'wrote {path}  ({size/1e6:.2f} MB)  {len(plan)} tensors')

    # tied head: 使反量化后的 lm_head == 反量化后的 embed
    dq_sd['lm_head.weight'] = dq_sd['model.embed_tokens.weight']

    # Golden: 加载反量化权重, forward 固定 prompt, 存最后位置 logits
    # 用 MHA 配置构建 (k/v 已转换为 q_heads 头), 与固件 llm.h 的 forward 一致
    gold_cfg = MiniMindConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        use_ple=True, ple_dim=cfg.ple_dim,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_attention_heads,  # MHA: kv_heads = q_heads
    )
    gold = MiniMindForCausalLM(gold_cfg)
    gold.load_state_dict(dq_sd)
    gold.eval()
    prompt = [1, 500, 1000, 200, 42, 777, 13, 99]
    ids = torch.tensor([prompt])
    with torch.no_grad():
        logits = gold(ids).logits
    last = logits[0, -1].numpy().astype(np.float32)
    base = os.path.join(args.out_dir, f'{args.weight}_h{args.hidden_size}')
    np.savez(base + '_golden.npz', prompt=np.array(prompt, dtype=np.int32), logits=last)
    with open(base + '_golden.txt', 'w') as gf:
        gf.write(f'{len(prompt)}\n')
        gf.write(' '.join(str(t) for t in prompt) + '\n')
        gf.write('\n'.join(f'{v:.6f}' for v in last) + '\n')
    top5 = last.argsort()[-5:][::-1]
    print(f'golden: prompt={prompt}')
    print(f'golden: last-pos top5 token ids = {top5.tolist()}')
    print(f'golden: logit range [{last.min():.3f}, {last.max():.3f}]')


if __name__ == '__main__':
    main()
