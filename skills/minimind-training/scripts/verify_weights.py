#!/usr/bin/env python3
"""
verify_weights.py — 训练产物质量检查

对训练产出的权重做完整性验证:
  1. 权重文件存在 + 大小
  2. MiniMindForCausalLM 加载 (missing/unexpected keys 统计 — 上游 init_model 用
     strict=False 静默容忍, 这里补上 AGENTS.md 要求的严格校验)
  3. forward 冒烟测试 (输入一批 token, 检查 logits 形状与数值有限性)

用法:
  python scripts/verify_weights.py --weight out/email_sft_dense_h256_256.pth \
      --hidden_size 256 --num_hidden_layers 6 [--use_ple] [--ple_dim 96]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="权重路径")
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_hidden_layers", type=int, default=6)
    ap.add_argument("--use_ple", action="store_true", help="手动指定 PLE; 默认按文件名 _ple 后缀自动推断")
    ap.add_argument("--ple_dim", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = ap.parse_args()

    # 自动推断 PLE: 权重文件名含 _ple 后缀则视为 PLE 权重
    if not args.use_ple and "_ple" in os.path.basename(args.weight):
        args.use_ple = True
        print(f"🔎 自动检测: {os.path.basename(args.weight)} 为 PLE 权重 (--use_ple)")

    report = {"weight": args.weight, "exists": os.path.exists(args.weight), "size_mb": None,
              "load_ok": False, "missing": None, "unexpected": None, "forward_ok": False,
              "param_budget": None}

    if not report["exists"]:
        print(json.dumps(report) if args.json else f"❌ 权重不存在: {args.weight}")
        sys.exit(1)

    report["size_mb"] = round(os.path.getsize(args.weight) / 1024**2, 2)

    cfg = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    if args.use_ple:
        cfg.use_ple = True
        cfg.ple_dim = args.ple_dim

    model = MiniMindForCausalLM(cfg)
    state = torch.load(args.weight, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    report["missing"] = len(result.missing_keys)
    report["unexpected"] = len(result.unexpected_keys)
    report["load_ok"] = report["missing"] == 0 and report["unexpected"] == 0

    # PLE 参数预算 (core/table/stream 三层)
    if args.use_ple:
        budget = model.param_budget()
        report["param_budget"] = {k: int(v) for k, v in budget.items()}
        report["param_budget"]["table_mb"] = round(budget["table"] * 4 / 1024**2, 2)

    if not report["load_ok"]:
        print(json.dumps(report) if args.json else
              f"⚠️ 部分加载: missing={report['missing']} unexpected={report['unexpected']} "
              f"(PLE 模式请确认 --use_ple)")
        if not args.json:
            for k in result.missing_keys[:5]:
                print(f"  missing: {k}")
            for k in result.unexpected_keys[:5]:
                print(f"  unexpected: {k}")

    if args.use_ple and report["param_budget"]:
        b = report["param_budget"]
        print(f"  PLE 参数预算: core={b['core']/1e6:.2f}M table={b['table']/1e6:.2f}M "
              f"stream={b['stream']/1e6:.2f}M total={b['total']/1e6:.2f}M")

    model.eval()
    with torch.no_grad():
        try:
            input_ids = torch.randint(0, 6400, (1, args.seq_len))
            logits = model(input_ids).logits
            finite = bool(torch.isfinite(logits).all())
            report["forward_ok"] = finite
            report["logits_shape"] = list(logits.shape)
            report["logits_std"] = float(logits.float().std())
            if args.json:
                print(json.dumps(report))
            else:
                status = "✅" if (report["load_ok"] and finite) else "❌"
                print(f"{status} {os.path.basename(args.weight)} "
                      f"({report['size_mb']}MB, missing={report['missing']}, "
                      f"unexpected={report['unexpected']}, logits_std={report['logits_std']:.4f})")
        except Exception as e:
            report["forward_ok"] = False
            report["error"] = str(e)[:200]
            print(json.dumps(report) if args.json else f"❌ forward 失败: {str(e)[:200]}")
            sys.exit(1)

    sys.exit(0 if (report["load_ok"] and report["forward_ok"]) else 1)


if __name__ == "__main__":
    main()
