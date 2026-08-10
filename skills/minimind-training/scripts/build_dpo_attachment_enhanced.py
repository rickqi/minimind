#!/usr/bin/env python3
"""
build_dpo_attachment_enhanced.py — 附件增强 DPO 数据构建 (B 方案核心)

利用附件富矿 (raw/*.md) 完善 DPO 偏好对:
1. 加载附件索引 (out/attachment_index.pkl, 由 build_attachment_index.py 构建)
2. 对 DPO 样本, 用 thread_id 查索引 → attachment_dir → raw/ 镜像 .md
3. 注入附件全文到 user_msg (top-2 × 2000 字符)
4. 输出增强版 DPO 数据

用法:
  python build_dpo_attachment_enhanced.py \
      --src /home/EmailAgent/data/training_data/split/dpo_train.jsonl \
      --index out/attachment_index.pkl \
      --out skills/minimind-training/dataset/dpo_email_attach.jsonl
"""
import argparse
import glob
import json
import os
import pickle

ATTACH_LIMIT = 2      # top-N 附件
ATTACH_CHARS = 2000   # 每附件截断字符
MAX_USER = 4000       # user_msg 总长上限


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_tid(tid: str) -> str:
    """dpo thread_id 带 _数字 后缀 (thread 序号), 剥离后与 conversation_id 匹配"""
    if "_" in tid:
        base, _, suffix = tid.rpartition("_")
        if suffix.isdigit():
            return base
    return tid


def find_attach_md(attachment_dir: str):
    raw_dir = attachment_dir.replace("downloaded_emails", "raw")
    return glob.glob(os.path.join(raw_dir, "*.md")) + glob.glob(os.path.join(raw_dir, "**", "*.md"), recursive=True)


def read_attach_md(md_path: str, max_chars: int) -> str:
    try:
        with open(md_path, encoding="utf-8") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def inject_attachment(user_msg: str, attach_mds) -> str:
    if not attach_mds:
        return user_msg
    parts = [f"[附件: {os.path.basename(md)}]\n{read_attach_md(md, ATTACH_CHARS)}"
             for md in attach_mds]
    injected = "\n\n".join(parts[:ATTACH_LIMIT])
    full = f"{user_msg}\n\n[附件内容]\n{injected}"
    return full[:MAX_USER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--index", default="out/attachment_index.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idx = load_index(args.index)
    print(f"附件索引: {len(idx)} 个 conversation_id")

    n_total, n_enhanced, n_missing = 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.src, encoding="utf-8"):
            d = json.loads(line)
            n_total += 1
            tid = normalize_tid(d.get("thread_id", ""))
            attach_dir = idx.get(tid, "")
            if not attach_dir:
                n_missing += 1
                continue
            attach_mds = find_attach_md(attach_dir)
            if not attach_mds:
                n_missing += 1
                continue
            for pair in (d["chosen"], d["rejected"]):
                for msg in pair:
                    if msg["role"] == "user":
                        msg["content"] = inject_attachment(msg["content"], attach_mds)
            d["enhanced"] = True
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_enhanced += 1

    print(f"结果: {n_enhanced}/{n_total} 对附件增强 ({n_enhanced/max(n_total,1)*100:.1f}%)")
    print(f"  无附件索引: {n_missing} 对")
    print(f"  输出: {args.out}")


if __name__ == "__main__":
    main()
