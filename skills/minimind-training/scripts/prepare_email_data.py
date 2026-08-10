#!/usr/bin/env python3
"""
prepare_email_data.py — EmailAgent 训练数据 → MiniMind 标准格式预处理

背景:
  EmailAgent 数据管线产物 (sft_tasks/sft_threads/dpo/pretrain) 格式与 MiniMind 数据类基本兼容,
  但 SFT 数据带多余字段 (task_type/thread_id/turn_count/time_span), 会被 SFTDataset 的
  严格 Features schema 拒绝 (datasets.table.CastError: column names don't match)。
  本脚本剥离多余字段, 输出 MiniMind 可直接消费的标准格式。

输出 (写入 skill 内 dataset/ 目录, 训练时用 --data_path 指向):
  dataset/sft_email_tasks.jsonl    纯 {"conversations": [...]}
  dataset/sft_email_threads.jsonl  纯 {"conversations": [...]}
  dataset/sft_email_mixed.jsonl    tasks + threads 合并 (主训练集)
  dataset/dpo_email.jsonl          原样拷贝 (DPO 无 Features 限制, 但统一走此入口)
  dataset/pretrain_email.jsonl     原样拷贝 {"text": ...}
  合并去重 + 报告打印

用法:
  python scripts/prepare_email_data.py
  python scripts/prepare_email_data.py --src /path/to/training_data --out dataset
"""
import argparse
import json
import os
import random
import shutil
import sys

SRC_DEFAULT = "/home/EmailAgent/data/training_data"

# SFTDataset 的 Features schema 只接受这些字段 (见 dataset/lm_dataset.py:63)
SFT_ALLOWED_KEYS = {"role", "content", "reasoning_content", "tools", "tool_calls"}


def strip_conversation(msg: dict) -> dict:
    """剥离 conversation 消息中的多余字段, 只保留 schema 允许的 key"""
    return {k: msg[k] for k in SFT_ALLOWED_KEYS if k in msg}


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_sft(src_path: str, out_path: str) -> tuple:
    """SFT 数据: 剥离多余顶层字段 + 消息字段"""
    total, dropped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in load_jsonl(src_path):
            total += 1
            conv = sample.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                dropped += 1
                continue
            cleaned = [strip_conversation(m) for m in conv]
            if not any(m.get("role") == "assistant" and m.get("content") for m in cleaned):
                dropped += 1
                continue
            f.write(json.dumps({"conversations": cleaned}, ensure_ascii=False) + "\n")
    return total, total - dropped


def copy_raw(src_path: str, out_path: str) -> int:
    """DPO/pretrain: 原样拷贝 (格式已兼容)"""
    shutil.copy(src_path, out_path)
    return sum(1 for _ in open(src_path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="EmailAgent 数据 → MiniMind 标准格式")
    ap.add_argument("--src", default=SRC_DEFAULT, help="EmailAgent training_data 目录")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"),
                    help="输出目录 (默认 skill 内 dataset/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    print("=" * 60)
    print("EmailAgent 数据预处理")
    print(f"  源目录: {args.src}")
    print(f"  输出目录: {args.out}")
    print("=" * 60)

    report = {}

    # 0. 优先使用 EmailAgent split/ 切分 (sft_train = 全部任务合并训练集)
    src_sft_train = os.path.join(args.src, "split", "sft_train.jsonl")
    out_sft_train = os.path.join(args.out, "sft_email_train_full.jsonl")
    if os.path.exists(src_sft_train):
        total, kept = convert_sft(src_sft_train, out_sft_train)
        report["sft_email_train_full"] = {"total": total, "kept": kept}
        print(f"  sft_train (split/): {kept}/{total} 条 → {out_sft_train}")

    # 1. SFT tasks
    src_tasks = os.path.join(args.src, "sft_tasks", "sft_email_tasks.jsonl")
    out_tasks = os.path.join(args.out, "sft_email_tasks.jsonl")
    if os.path.exists(src_tasks):
        total, kept = convert_sft(src_tasks, out_tasks)
        report["sft_tasks"] = {"total": total, "kept": kept}
        print(f"  sft_tasks: {kept}/{total} 条 (剥离 task_type 字段) → {out_tasks}")

    # 2. SFT threads
    src_threads = os.path.join(args.src, "sft_threads", "sft_email_threads.jsonl")
    out_threads = os.path.join(args.out, "sft_email_threads.jsonl")
    if os.path.exists(src_threads):
        total, kept = convert_sft(src_threads, out_threads)
        report["sft_threads"] = {"total": total, "kept": kept}
        print(f"  sft_threads: {kept}/{total} 条 (剥离 thread_id/turn_count/time_span) → {out_threads}")

    # 3. 合并 tasks + threads (主 SFT 训练集)
    out_mixed = os.path.join(args.out, "sft_email_mixed.jsonl")
    if os.path.exists(out_tasks) and os.path.exists(out_threads):
        seen, n = set(), 0
        with open(out_mixed, "w", encoding="utf-8") as f:
            for path in (out_tasks, out_threads):
                for line in open(path, encoding="utf-8"):
                    key = line.strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(line)
                    n += 1
        report["sft_email_mixed"] = n
        print(f"  sft_email_mixed: {n} 条 (tasks+threads 合并去重) → {out_mixed}")

    # 4. DPO (原样拷贝)
    src_dpo = os.path.join(args.src, "dpo", "dpo_email.jsonl")
    out_dpo = os.path.join(args.out, "dpo_email.jsonl")
    if os.path.exists(src_dpo):
        n = copy_raw(src_dpo, out_dpo)
        report["dpo_email"] = n
        print(f"  dpo_email: {n} 条 (原样拷贝) → {out_dpo}")

    # 5. Pretrain (原样拷贝)
    src_pretrain = os.path.join(args.src, "pretrain", "pretrain_email.jsonl")
    out_pretrain = os.path.join(args.out, "pretrain_email.jsonl")
    if os.path.exists(src_pretrain):
        n = copy_raw(src_pretrain, out_pretrain)
        report["pretrain_email"] = n
        print(f"  pretrain_email: {n} 条 (原样拷贝) → {out_pretrain}")

    # 6. 质量校验: 用 SFTDataset 实测可加载性
    print("-" * 60)
    print("质量校验 (SFTDataset 可加载性):")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".."))
        from transformers import AutoTokenizer
        from dataset.lm_dataset import SFTDataset

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "model"))
        for name in ("sft_email_tasks", "sft_email_threads", "sft_email_mixed"):
            p = os.path.join(args.out, f"{name}.jsonl")
            if os.path.exists(p):
                ds = SFTDataset(p, tokenizer, max_length=128)
                print(f"  ✅ {name}: SFTDataset 加载成功, {len(ds)} 条")
    except ImportError:
        print("  ⚠️ 无法导入 minimind dataset (跳过加载校验, 请确认在项目根运行)")
    except Exception as e:
        print(f"  ⚠️ 加载校验失败: {type(e).__name__}: {str(e)[:200]}")

    print("=" * 60)
    print("预处理完成:", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
