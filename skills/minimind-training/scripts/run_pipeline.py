#!/usr/bin/env python3
"""
run_pipeline.py — MiniMind 训练管道统一入口

一键执行完整训练流程: 环境检查 → 数据预处理 → 训练 → 质量检查 → 评估。
解决单步脚本分散、依赖缺失、tee 目录未创建等流程问题。

用法:
  python run_pipeline.py --mode 1 [--stage all|env|data|train|verify|eval]
  python run_pipeline.py --mode 2 [--data 2000] [--epochs 3]
  --mode 1 = 手段1 默认 Dense (use_ple=0)
  --mode 2 = 手段2 PLE (use_ple=1)

每个 stage 可独立运行, 已完成的 stage 默认跳过 (可 --force 重跑)。
"""
import argparse
import os
import subprocess
import sys
import time

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(SKILL_DIR, "..", "..")
SCRIPTS = os.path.join(SKILL_DIR, "scripts")
DATASET = os.path.join(SKILL_DIR, "dataset")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")

MODE_CFG = {
    1: {"name": "手段1 默认 Dense", "use_ple": False, "suffix": ""},
    2: {"name": "手段2 PLE", "use_ple": True, "suffix": "_ple"},
}


def run(cmd, step):
    print(f"\n{'='*70}\n▶ [{step}]\n{'='*70}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"❌ [{step}] 失败 (exit={r.returncode}, {elapsed:.0f}s)")
        sys.exit(r.returncode)
    print(f"✅ [{step}] 完成 ({elapsed:.0f}s)")
    return r


def stage_env():
    print("\n=== 环境检查 ===")
    ok = True
    for mod in ("torch", "transformers", "datasets", "tokenizers"):
        try:
            m = __import__(mod)
            print(f"  ✅ {mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  ❌ {mod} 缺失")
            ok = False
    try:
        import torch
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)} / CUDA={torch.cuda.is_available()}")
    except Exception as e:
        print(f"  ⚠️ GPU 检测失败: {e}")
    tok = os.path.join(PROJECT_ROOT, "model", "tokenizer.json")
    print(f"  {'✅' if os.path.exists(tok) else '❌'} tokenizer: {tok}")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  ✅ out 目录: {OUT_DIR}")
    if not ok:
        print("\n❌ 缺少依赖, 请先: pip install datasets transformers")
        sys.exit(1)


def stage_data(force=False):
    src = "/home/EmailAgent/data/training_data"
    if not os.path.exists(src):
        print(f"⚠️ 数据源不存在: {src}, 跳过预处理")
        return
    out_mixed = os.path.join(DATASET, "sft_email_mixed.jsonl")
    if os.path.exists(out_mixed) and not force:
        print(f"✅ 数据已就绪: {out_mixed} (跳过, --force 重跑)")
        return
    run([sys.executable, os.path.join(SCRIPTS, "prepare_email_data.py"), "--src", src], "数据预处理")


def stage_train(mode, data, epochs, force=False):
    cfg = MODE_CFG[mode]
    weight = f"email_sft_{'ple' if mode == 2 else 'dense'}_h256_256{cfg['suffix']}.pth"
    weight_path = os.path.join(OUT_DIR, weight)
    if os.path.exists(weight_path) and not force:
        print(f"✅ 权重已存在: {weight_path} (跳过, --force 重跑)")
        return
    script = os.path.join(SCRIPTS, f"train_mode{mode}_{'ple' if mode == 2 else 'default'}_sft.sh")
    run(["bash", script, data, str(epochs), "256", "6"], f"训练 手段{mode} ({cfg['name']})")


def stage_verify(mode, force=False):
    cfg = MODE_CFG[mode]
    weight = f"email_sft_{'ple' if mode == 2 else 'dense'}_h256_256{cfg['suffix']}.pth"
    weight_path = os.path.join(OUT_DIR, weight)
    if not os.path.exists(weight_path):
        print(f"⚠️ 权重不存在: {weight_path}, 跳过验证")
        return
    cmd = [sys.executable, os.path.join(SCRIPTS, "verify_weights.py"),
           "--weight", weight_path, "--hidden_size", "256", "--num_hidden_layers", "6"]
    if mode == 2:
        cmd += ["--use_ple", "--ple_dim", "96"]
    run(cmd, f"质量检查 手段{mode}")


def stage_eval(mode, force=False):
    cfg = MODE_CFG[mode]
    weight = f"email_sft_{'ple' if mode == 2 else 'dense'}_h256_256{cfg['suffix']}.pth"
    weight_path = os.path.join(OUT_DIR, weight)
    if not os.path.exists(weight_path):
        print(f"⚠️ 权重不存在: {weight_path}, 跳过评估")
        return
    cmd = [sys.executable, os.path.join(SCRIPTS, "eval_email.py"),
           "--weight", weight_path, "--hidden_size", "256", "--num_hidden_layers", "6",
           "--per_type", "1", "--max_new_tokens", "40"]
    if mode == 2:
        cmd += ["--use_ple", "--ple_dim", "96"]
    run(cmd, f"评估 手段{mode}")


def main():
    ap = argparse.ArgumentParser(description="MiniMind 训练管道")
    ap.add_argument("--mode", type=int, choices=[1, 2], required=True, help="1=默认Dense, 2=PLE")
    ap.add_argument("--stage", default="all", choices=["all", "env", "data", "train", "verify", "eval"])
    ap.add_argument("--data", default="sft_email_mixed_400.jsonl", help="训练数据文件")
    ap.add_argument("--epochs", type=int, default=2, help="训练轮数")
    ap.add_argument("--force", action="store_true", help="强制重跑已完成的 stage")
    args = ap.parse_args()

    cfg = MODE_CFG[args.mode]
    print(f"\n=== MiniMind 训练管道 | {cfg['name']} ===")

    stages = {
        "env": lambda: stage_env(),
        "data": lambda: stage_data(args.force),
        "train": lambda: stage_train(args.mode, args.data, args.epochs, args.force),
        "verify": lambda: stage_verify(args.mode, args.force),
        "eval": lambda: stage_eval(args.mode, args.force),
    }
    order = ["env", "data", "train", "verify", "eval"]
    targets = order if args.stage == "all" else [args.stage]

    # env 是前置, 任何 stage 都先跑
    if args.stage != "env":
        stage_env()

    for s in targets:
        if s == "env":
            continue
        stages[s]()

    print(f"\n=== 管道完成 ({cfg['name']}) ===")


if __name__ == "__main__":
    main()
