#!/usr/bin/env python3
"""
build_email_raft.py — EmailAgent 邮件域 RAFT 数据构建

医学 RAFT (build_medical_raft.py) 依赖知识库 QA 对做证据检索, 邮件场景无独立证据源。
邮件域 RAFT 适配: 证据 = 邮件原文本身 (作为"参考资料"注入 system), 训练模型基于给定
邮件执行任务 (分类/摘要/回复)。这模拟真实部署: 用户给一封邮件 → 模型基于邮件内容回答。

输出 (默认 skill dataset/):
  sft_email_raft.jsonl — {"conversations": [{system: 参考资料}, {user: 任务}, {assistant: 答案}]}

用法:
  python build_email_raft.py [--src sft_email_tasks.jsonl] [--out sft_email_raft.jsonl]
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "sft_email_tasks.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "sft_email_raft.jsonl"))
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--no-evidence-ratio", type=float, default=0.3, help="无证据比例 (防 RAFT 遗忘内在能力)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    samples = []
    for line in open(args.src, encoding="utf-8"):
        d = json.loads(line)
        conv = d.get("conversations", [])
        if len(conv) < 2:
            continue
        user_msg = conv[0].get("content", "")
        assistant_msg = conv[-1].get("content", "")
        if len(user_msg) < 50 or len(assistant_msg) < 10:
            continue
        samples.append((user_msg, assistant_msg))

    # 从邮件内容提取"参考资料" (用户消息中的邮件正文部分, 去掉任务指令前缀)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for user_msg, assistant_msg in samples:
            if n >= args.max_samples:
                break
            # 提取邮件正文: 去掉 "请...：" 任务前缀, 保留正文
            lines = user_msg.split("\n")
            mail_body = "\n".join(lines[1:]) if len(lines) > 1 else user_msg
            if len(mail_body) < 20:
                continue

            if random.random() < args.no_evidence_ratio:
                # 无证据样本: 直接给任务 (保内在能力)
                conv = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            else:
                # 证据样本: system 注入邮件正文作为参考资料
                conv = [
                    {"role": "system", "content": f"以下是需要处理的邮件正文：\n{mail_body}"},
                    {"role": "user", "content": f"请根据上述邮件完成以下任务：\n{lines[0]}"},
                    {"role": "assistant", "content": assistant_msg},
                ]
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
            n += 1

    print(f"邮件域 RAFT 数据: {n} 条 → {args.out}")
    print(f"  配置: 无证据比例 {args.no_evidence_ratio}, 最大 {args.max_samples}")


if __name__ == "__main__":
    main()
