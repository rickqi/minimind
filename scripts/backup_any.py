#!/usr/bin/env python3
"""
backup_any.py — 通用文件/目录备份到腾讯云 COS

用于备份大体积训练数据 (如 EmailAgent 169 万预训练语料 3.5GB) 到 COS,
避免本地丢失。压缩用 BZIP2 (高压缩比), 时间戳命名不覆盖。

配置: .env.cosine (COS_SECRET_ID/KEY/BUCKET/REGION)

用法:
  python backup_any.py --src /path/to/data.jsonl [--name pretrain_v2] [--list]
"""
import argparse
import os
import sys
import zipfile
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env.cosine")
COS_PREFIX = "backups/email-pretrain/"


def load_cos_config():
    cfg = {}
    if not os.path.exists(ENV_FILE):
        sys.exit(f"[ERR] 缺少 {ENV_FILE} (COS_SECRET_ID/KEY/BUCKET/REGION)")
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    required = ["COS_SECRET_ID", "COS_SECRET_KEY", "COS_BUCKET", "COS_REGION"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        sys.exit(f"[ERR] .env.cosine 缺少: {missing}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="要备份的文件或目录")
    ap.add_argument("--name", default="data", help="备份名前缀")
    ap.add_argument("--list", action="store_true", help="列出已有备份")
    ap.add_argument("--keep-local", action="store_true", help="保留本地压缩包")
    ap.add_argument("--dry-run", action="store_true", help="只打包不上传")
    args = ap.parse_args()

    from qcloud_cos import CosConfig, CosS3Client
    cfg = load_cos_config()
    client = CosS3Client(CosConfig(Region=cfg["COS_REGION"],
                                   SecretId=cfg["COS_SECRET_ID"],
                                   SecretKey=cfg["COS_SECRET_KEY"]))

    if args.list:
        resp = client.list_objects(Bucket=cfg["COS_BUCKET"], Prefix=COS_PREFIX)
        total = 0
        for c in resp.get("Contents", []):
            size = int(c["Size"])
            total += size
            print(f'  {c["Key"]}  {size/1024/1024:.1f}MB')
        print(f"  共 {len(resp.get('Contents', []))} 个备份, 总计 {total/1024/1024/1024:.2f}GB")
        return

    if not args.src or not os.path.exists(args.src):
        sys.exit(f"[ERR] 源不存在: {args.src}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{args.name}_{ts}.zip"
    zip_path = f"/tmp/{backup_name}"
    cos_key = f"{COS_PREFIX}{backup_name}"

    print(f"[1/3] 打包 {args.src} → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_BZIP2) as zf:
        if os.path.isdir(args.src):
            for root, _, files in os.walk(args.src):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.relpath(full, args.src))
        else:
            zf.write(args.src, os.path.basename(args.src))
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"  压缩包: {size_mb:.1f}MB")

    if args.dry_run:
        print(f"[2/3] [DRY-RUN] 跳过上传")
    else:
        print(f"[2/3] 上传 → cos://{cfg['COS_BUCKET']}/{cos_key}")
        client.put_object(Bucket=cfg["COS_BUCKET"], Key=cos_key, Body=open(zip_path, "rb"))
        print("  上传完成")

    if not args.keep_local:
        os.remove(zip_path)
        print(f"[3/3] 已清理本地压缩包 (--keep-local 保留)")


if __name__ == "__main__":
    main()
