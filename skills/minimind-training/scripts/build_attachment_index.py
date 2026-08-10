#!/usr/bin/env python3
"""
build_attachment_index.py — 附件索引构建 (conversation_id → attachment_dir)

parsed.json 总计 21GB/83 文件, 无法全量加载内存。
本脚本用 json.load 逐文件提取 (conv_id → attachment_dir) 轻量映射,
持久化为 pickle 索引 (out/attachment_index.pkl), 供附件增强脚本复用。

用法:
  python build_attachment_index.py [--out out/attachment_index.pkl] [--limit N(仅前N文件, 测试用)]
"""
import argparse
import glob
import json
import os
import pickle
import time

PARSED_GLOB = "/home/EmailAgent/data/downloaded_emails/*/analysis/monthly/*_parsed.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/attachment_index.pkl")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 parsed.json (测试)")
    args = ap.parse_args()

    pfs = sorted(glob.glob(PARSED_GLOB))
    if args.limit:
        pfs = pfs[: args.limit]
    print(f"处理 {len(pfs)} 个 parsed.json ({sum(os.path.getsize(f) for f in pfs)/1024**3:.1f}GB)")

    idx = {}
    t0 = time.time()
    for i, pf in enumerate(pfs):
        t1 = time.time()
        try:
            with open(pf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [{i+1}/{len(pfs)}] ⚠️ {os.path.basename(pf)}: {e}")
            continue
        for e in data.get("emails", []):
            cid = e.get("conversation_id")
            if cid and e.get("has_attachments") and e.get("attachment_dir"):
                idx[cid] = e["attachment_dir"]
        print(f"  [{i+1}/{len(pfs)}] {os.path.basename(pf)}: "
              f"{len([v for v in [e for e in data.get('emails',[]) if e.get('has_attachments')]])} 含附件 | "
              f"累计 {len(idx)} 条 | {time.time()-t1:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(idx, f)
    print(f"\n索引完成: {len(idx)} 个含附件 conversation_id → {args.out}")
    print(f"总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
