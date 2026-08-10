#!/usr/bin/env python3
"""
eval_email.py — EmailAgent 模型问答评估

用固定测试集评估训练产出的模型, 打印每个问题的回答。
测试集从 EmailAgent sft 数据中抽取 (分类/摘要/回复草稿各若干)。

用法:
  python scripts/eval_email.py --weight out/email_sft_dense_h256_256.pth \
      --hidden_size 256 --num_hidden_layers 6 [--use_ple] [--ple_dim 96]
"""
import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")


def build_test_set(src_path: str, per_type: int = 2, seed: int = 42, all_samples: bool = False) -> list:
    """从 SFT 数据中按 task_type 抽取测试问题; all_samples=True 时不抽样 (独立测试集)"""
    random.seed(seed)
    by_type = {}
    for line in open(src_path, encoding="utf-8"):
        d = json.loads(line)
        conv = d.get("conversations", [])
        if not conv or conv[0].get("role") != "user":
            continue
        q = conv[0]["content"][:200]
        t = d.get("task_type", "unknown")
        by_type.setdefault(t, []).append(q)
    tests = []
    for t, qs in by_type.items():
        picked = qs if all_samples else random.sample(qs, min(per_type, len(qs)))
        for q in picked:
            tests.append({"task_type": t, "question": q})
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_hidden_layers", type=int, default=6)
    ap.add_argument("--use_ple", action="store_true")
    ap.add_argument("--ple_dim", type=int, default=96)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "sft_email_tasks.jsonl"))
    ap.add_argument("--per_type", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="评估全部样本 (不抽样, 用于独立测试集)")
    ap.add_argument("--device", default="cpu", help="评估设备: cpu (默认, ROCm 生成稳定) / cuda")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "model"))
    cfg = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    if args.use_ple:
        cfg.use_ple = True
        cfg.ple_dim = args.ple_dim

    model = MiniMindForCausalLM(cfg)
    state = torch.load(args.weight, map_location="cpu")
    r = model.load_state_dict(state, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print(f"⚠️ 权重部分加载: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")
    # 默认 CPU: ROCm (AMD 890M) 上 cuda 推理不可靠; 真 NVIDIA GPU 可 --device cuda
    device = args.device if args.device in ("cuda", "cpu") else "cpu"
    model = model.to(device)
    model.eval()

    tests = build_test_set(args.data, per_type=args.per_type, all_samples=args.all)
    print(f"=== 评估 {os.path.basename(args.weight)} ({device}) ===")
    print(f"测试集: {len(tests)} 问 ({args.data})\n")

    results = []
    for i, t in enumerate(tests):
        messages = [{"role": "user", "content": t["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            # do_sample=False: 规避 ROCm (AMD 890M + torch rocm) 上 multinomial 长序列采样的死循环 bug
            # use_cache=False: 规避同环境 KV cache 生成卡死 (二者结合在此环境可稳定生成)
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                                 temperature=0.7, top_p=1.0, top_k=0, do_sample=False,
                                 use_cache=False)
        answer = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        results.append({"task_type": t["task_type"], "question": t["question"][:80], "answer": answer[:150]})
        print(f"[{i+1}] ({t['task_type']})")
        print(f"  Q: {t['question'][:80]}")
        print(f"  A: {answer[:150]}")
        print()

    # 分类准确率 (仅对分类任务, 核心标签前缀匹配)
    CORE_LABELS = ["项目讨论", "日常协作", "合同审核", "权限管理", "脱敏管理", "监控告警", "数据出境"]
    cls = [t for t in results if "分类" in t["question"]]
    if cls:
        correct = sum(1 for t in cls if any(t["answer"].strip().startswith(lb) for lb in CORE_LABELS))
        print(f"\n📊 分类任务准确率: {correct}/{len(cls)} = {correct/len(cls)*100:.1f}% (核心标签: {CORE_LABELS})")

    return results


if __name__ == "__main__":
    main()
