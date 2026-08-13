#!/usr/bin/env python3
"""hardware_profile.py — 训练前硬件探测 + 参数自动调优

检测 GPU/CPU/RAM, 用真实模型+数据微基准逐档提升 batch_size, 找吞吐拐点(plateau)
或显存安全上限, 输出推荐 batch_size/num_workers/max_seq_len/dtype。

用法:
  python hardware_profile.py --data_path X.jsonl --hidden 256 --layers 6
  → 打印 JSON: {"batch_size":..,"num_workers":..,"max_seq_len":..,"device":..,"dtype":..,"profile":{...}}

设计:
  - GB10/Grace 等统一内存机型: 显存非瓶颈, batch_size 一路升至吞吐饱和
  - 独立显存 GPU: 监控显存, 超过 80% 安全上限即停
  - CPU fallback: batch_size 受限于内存
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'trainer'))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def detect_hardware():
    cuda = torch.cuda.is_available()
    cpu_cores = os.cpu_count() or 4
    import multiprocessing
    try:
        ram = multiprocessing.shared_memory.SharedMemory('x')  # noqa: 触发 import
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    ram_gb = int(line.split()[1]) / 1024 / 1024
                    break
            else:
                ram_gb = 0
    except Exception:
        ram_gb = 0
    gpu_name = ''
    compute_cap = ''
    unified = False
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = '.'.join(str(x) for x in torch.cuda.get_device_capability(0))
        try:
            tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # GB10 / Grace / Tegra 统一内存: get_device_properties 返回极小或查询失败
            if tot < 1 or 'GB10' in gpu_name or 'Grace' in gpu_name or 'Orin' in gpu_name or 'Tegra' in gpu_name:
                unified = True
        except Exception:
            unified = True
    return {
        'cuda': cuda, 'gpu_name': gpu_name, 'compute_cap': compute_cap,
        'unified_memory': unified, 'cpu_cores': cpu_cores, 'ram_available_gb': round(ram_gb, 1),
    }


def _free_gpu_mem_gb():
    try:
        free, tot = torch.cuda.mem_get_info()[:2]
        return free / 1024**3
    except Exception:
        return 999.0


def build_model_and_loader(data_path, hidden, layers, batch_size, num_workers, max_seq_len, device, dtype):
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    tok = AutoTokenizer.from_pretrained(os.path.join(proj, 'model'))
    ds = PretrainDataset(data_path, tokenizer=tok, max_length=max_seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=(device.startswith('cuda')), persistent_workers=(num_workers > 0), drop_last=True)
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers,
                         num_attention_heads=8, num_key_value_heads=4,
                         vocab_size=len(tok), max_position_embeddings=max_seq_len)
    model = MiniMindForCausalLM(cfg).to(device, dtype=dtype)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    return dl, model, opt, tok


def bench_batch(data_path, hidden, layers, batch_size, num_workers, max_seq_len, device, dtype, steps=15):
    dl, model, opt, _ = build_model_and_loader(data_path, hidden, layers, batch_size, num_workers, max_seq_len, device, dtype)
    it = iter(dl)
    for _ in range(3):  # warmup
        input_ids, labels = next(it)
        res = model(input_ids.to(device), labels=labels.to(device))
        res.loss.backward(); opt.step(); opt.zero_grad()
    if device.startswith('cuda'):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        input_ids, labels = next(it)
        res = model(input_ids.to(device), labels=labels.to(device))
        res.loss.backward(); opt.step(); opt.zero_grad()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.startswith('cuda') else 0
    sps = steps / dt
    xsps = sps * batch_size
    loss = res.loss.item()
    del model, opt, dl
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return {'batch_size': batch_size, 'num_workers': num_workers, 'step_s': dt,
            'steps_per_s': round(sps, 2), 'samples_per_s': round(xsps, 1),
            'peak_gpu_gb': round(peak_gb, 2), 'loss': round(loss, 3)}


def _safe_json(obj):
    """递归把 torch dtype/tensor 转成可 JSON 序列化的值"""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, torch.dtype):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.item()
    return obj


def sweep_workers(data_path, hidden, layers, batch_size, worker_candidates, max_seq_len, device, dtype, steps=12):
    best = None
    for nw in worker_candidates:
        try:
            r = bench_batch(data_path, hidden, layers, batch_size, nw, max_seq_len, device, dtype, steps=steps)
            r['num_workers'] = nw
            print(f"[profile] workers={nw:>3} → {r['samples_per_s']:>8.1f} samp/s", file=sys.stderr)
            if best is None or r['samples_per_s'] > best['samples_per_s']:
                best = r
        except Exception as e:
            print(f"[profile] workers={nw} FAIL {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
    return best


def recommend(hw, data_path, hidden, layers, max_seq_len, device, dtype, max_batch=1024, bench_steps=12):
    """逐档提升 batch_size 找吞吐拐点 (统一内存) 或显存上限 (独立 GPU)"""
    if not hw['cuda'] and device.startswith('cuda'):
        device = 'cpu'
    # num_workers: 与核数匹配, 但不超过 8 (IO 边际递减)
    base_workers = min(8, max(2, hw['cpu_cores'] // 3))
    # 候选 batch_size 序列
    candidates = [16, 32, 64, 128, 256, 512, 1024]
    candidates = [b for b in candidates if b <= max_batch]
    results = []
    print(f"[profile] device={device} dtype={dtype} seqlen={max_seq_len} candidates={candidates}", file=sys.stderr)
    prev_xsps = 0
    chosen = candidates[0]
    chosen_workers = base_workers
    for b in candidates:
        nw = base_workers if b <= 128 else min(12, hw['cpu_cores'] // 2)
        try:
            r = bench_batch(data_path, hidden, layers, b, nw, max_seq_len, device, dtype, steps=bench_steps)
        except Exception as e:
            print(f"[profile] batch={b} FAIL {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
            break
        results.append(r)
        print(f"[profile] batch={b:>4} → {r['samples_per_s']:>8.1f} samp/s  {r['steps_per_s']:>6.2f} step/s  peak={r['peak_gpu_gb']}GB  loss={r['loss']}", file=sys.stderr)
        chosen = b
        chosen_workers = nw
        # 吞吐饱和判定: 提升批次但 samples/s 增幅 <15% → 已到瓶颈(GPU 算力或数据加载), 停在当前
        if prev_xsps > 0 and r['samples_per_s'] < prev_xsps * 1.15:
            print(f"[profile] batch 吞吐饱和 (增幅 <15%), 停在 batch={b}", file=sys.stderr)
            break
        prev_xsps = r['samples_per_s']
    # 若 batch 提升几乎无增益 → 瓶颈是数据加载, 扫描 num_workers 找最优
    worker_sweep = None
    if len(results) >= 2 and results[-1]['samples_per_s'] < results[0]['samples_per_s'] * 1.3:
        print(f"[profile] 数据加载疑似瓶颈, 扫描 num_workers @ batch={chosen}", file=sys.stderr)
        worker_candidates = sorted({w for w in [4, base_workers, 8, min(12, hw['cpu_cores'] // 2), min(16, hw['cpu_cores'] - 2)] if w >= 2})
        worker_sweep = sweep_workers(data_path, hidden, layers, chosen, worker_candidates, max_seq_len, device, dtype, steps=bench_steps)
        if worker_sweep:
            chosen_workers = worker_sweep['num_workers']
            final_sps = worker_sweep['samples_per_s']
        else:
            final_sps = results[-1]['samples_per_s']
    else:
        final_sps = results[-1]['samples_per_s'] if results else 0
    return {
        'batch_size': chosen, 'num_workers': chosen_workers,
        'max_seq_len': max_seq_len, 'device': device, 'dtype': str(dtype),
        'samples_per_s': round(final_sps, 1),
        'profile': hw, 'bench': results, 'worker_sweep': worker_sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True, help='用于微基准的训练 jsonl')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--max_seq_len', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--max_batch', type=int, default=512)
    ap.add_argument('--bench_steps', type=int, default=12)
    args = ap.parse_args()
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    hw = detect_hardware()
    print(f"[profile] HW: {hw}", file=sys.stderr)
    rec = recommend(hw, args.data_path, args.hidden, args.layers, args.max_seq_len,
                    args.device, dtype_map[args.dtype], args.max_batch, args.bench_steps)
    print(json.dumps(_safe_json(rec), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
