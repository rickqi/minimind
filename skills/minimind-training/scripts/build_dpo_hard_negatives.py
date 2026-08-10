#!/usr/bin/env python3
"""
build_dpo_hard_negatives.py — on-policy 硬负样本 DPO 数据构建 (待执行任务 ③)

核心思路: DPO 模板负样本 (rejected) 不在策略分布 → 梯度信号弱。
本脚本用 SFT 模型对 prompt 采样多个回复, 用启发式/长度规则选出"较差"回复作为 rejected,
确保 rejected 是模型"可能生成但质量较低"的输出 (in-distribution hard negative)。

产出: dpo_email_hardneg.jsonl (chosen=原优质回复, rejected=模型采样较差回复)

用法:
  python build_dpo_hard_negatives.py \
      --src skills/minimind-training/dataset/dpo_email_attach.jsonl \
      --weight out/email_sft_dense_h256_256.pth \
      --out skills/minimind-training/dataset/dpo_email_hardneg.jsonl \
      --samples 2000
"""
import argparse
import json
import random

import torch
from transformers import AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")

# 硬负样本质量规则: 采样回复若命中这些"敷衍模式"则判为差 (宁可少选, 保证质量)
WEAK_PATTERNS = ["收到", "好的", "谢谢", "了解", "稍后", "尽快", "OK", "ok", "没问题"]


def is_weak_reply(text: str, chosen_len: int) -> bool:
    """判定采样回复是否为弱回复。

    长度偏置问题: 只判短回复 (len<15) 会让 rejected 恒短 → 长度比 >20x。
    改进: rejected 长度需与 chosen 接近 (0.5-1.5x), 同时内容空洞 (模板/重复/信息少)。
    """
    t = text.strip()
    if len(t) < 15:
        return False  # 太短不选 (避免长度偏置, 交给后续长度匹配)
    # 长度匹配: rejected 应在 chosen 的 0.5-1.5x 范围 (防止长度偏置)
    ratio = max(len(t), chosen_len) / max(min(len(t), chosen_len), 1)
    if ratio > 1.5:
        return False  # 长度差太大不选
    # 内容质量: 模板开头 / 重复 / 信息密度低 → 判为弱
    if any(t.startswith(p) for p in WEAK_PATTERNS):
        return True
    if len(set(t)) < 8:
        return True
    # 信息密度: 与 chosen 相比明显空洞
    if len(t) < chosen_len * 0.5:
        return True
    return False


def sample_reply(model, tokenizer, user_msg: str, device: str, num_samples: int) -> list:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg[:1500]}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    replies = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model.generate(
                ids, max_new_tokens=80, do_sample=True,
                temperature=0.9, top_p=0.9, use_cache=False,
            )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if text and len(text) > 5:
                replies.append(text)
    return replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="附件增强 DPO 数据 (或原 dpo)")
    ap.add_argument("--weight", default=os.path.join(PROJECT_ROOT, "out", "email_sft_dense_h256_256.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2000, help="生成硬负样本的对数")
    ap.add_argument("--num-samples", type=int, default=4, help="每 prompt 采样回复数")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"加载 SFT 模型: {args.weight}")
    cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=6)
    model = MiniMindForCausalLM(cfg)
    model.load_state_dict(torch.load(args.weight, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "model"))

    src_lines = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    random.seed(42)
    picked = random.sample(src_lines, min(args.samples, len(src_lines)))

    n_ok, n_skip = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, d in enumerate(picked):
            user_msg = d["chosen"][0]["content"]
            chosen = d["chosen"][-1]["content"]
            replies = sample_reply(model, tokenizer, user_msg, args.device, args.num_samples)
            weak = [r for r in replies if is_weak_reply(r, len(chosen))]
            if not weak:
                n_skip += 1
                continue
            rejected = weak[0]
            out = {
                "chosen": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": chosen},
                ],
                "rejected": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": rejected},
                ],
                "source": "hardneg",
                "thread_id": d.get("thread_id", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(picked)}] 生成 {n_ok} 对, 跳过 {n_skip}")

    print(f"\n硬负样本 DPO 数据: {n_ok} 对 → {args.out}")
    print(f"  跳过 (模型未产生弱回复): {n_skip}")


if __name__ == "__main__":
    main()
