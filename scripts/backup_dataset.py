#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset 备份脚本: 压缩本地 dataset/ -> 时间戳 zip -> 上传腾讯云 COS

特性:
- 每次备份生成唯一时间戳 (YYYYMMDD_HHMMSS), 不覆盖历史备份
- ZIP_BZIP2 压缩 (高压缩比, 比 deflate 高 ~40%)
- 备份路径 backups/minimind/ (与其他体系 data/training_data, email_knowledge 隔离)
- 密钥从 .env.cosine 读取 (gitignored, 不入库)
- 支持 --keep-local 保留本地压缩包 (默认删除)

用法:
  python scripts/backup_dataset.py                 # 备份全部 dataset/
  python scripts/backup_dataset.py --list          # 列出已有备份
  python scripts/backup_dataset.py --keep-local    # 保留本地压缩包
  python scripts/backup_dataset.py --dry-run       # 只打包不上传 (测试)
"""
import argparse
import datetime
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
ENV_FILE = os.path.join(PROJECT_ROOT, '.env.cosine')
# 前缀带 minimind, 存到 backups/minimind/ 子目录, 避免与桶内其他备份冲突
BACKUP_PREFIX = 'minimind-dataset-backup'
COS_PREFIX = 'backups/minimind/'
COMPRESSION = zipfile.ZIP_BZIP2  # 高压缩比


def load_cos_config():
    """从 .env.cosine 加载 COS 配置."""
    if not os.path.exists(ENV_FILE):
        sys.exit(f'[ERR] 未找到密钥文件 {ENV_FILE} (请先创建)')
    cfg = {}
    with open(ENV_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    required = ['COS_SECRET_ID', 'COS_SECRET_KEY', 'COS_BUCKET', 'COS_REGION']
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        sys.exit(f'[ERR] .env.cosine 缺少配置: {missing}')
    return cfg


def make_archive(backup_name, dry_run=False):
    """打包 dataset/ 为 zip (BZIP2 高压缩), 返回 (压缩包路径, 总大小MB)."""
    out_path = os.path.join(PROJECT_ROOT, f'{backup_name}.zip')
    if dry_run:
        total = sum(os.path.getsize(os.path.join(DATASET_DIR, f))
                    for f in os.listdir(DATASET_DIR)
                    if os.path.isfile(os.path.join(DATASET_DIR, f)))
        return out_path, total / 1e6
    print(f'[1/3] 打包 {DATASET_DIR} -> {out_path} (BZIP2 高压缩)')
    total = 0
    with zipfile.ZipFile(out_path, 'w', COMPRESSION, compresslevel=9) as zf:
        for f in sorted(os.listdir(DATASET_DIR)):
            p = os.path.join(DATASET_DIR, f)
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.join('dataset', f))
                total += os.path.getsize(p)
    print(f'      完成: {os.path.getsize(out_path) / 1e6:.1f}MB (源 {total / 1e6:.0f}MB, 压缩比 {total / max(os.path.getsize(out_path), 1):.1f}x)')
    return out_path, total / 1e6


def upload_to_cos(cfg, backup_name, local_path, dry_run=False):
    """上传到 COS bucket 的 backups/minimind/ 目录."""
    from qcloud_cos import CosConfig, CosS3Client
    if dry_run:
        print(f'[3/3] [DRY-RUN] 模拟上传 {backup_name} -> cos://{cfg["COS_BUCKET"]}/{COS_PREFIX}')
        return True
    config = CosConfig(Region=cfg['COS_REGION'],
                       SecretId=cfg['COS_SECRET_ID'],
                       SecretKey=cfg['COS_SECRET_KEY'])
    client = CosS3Client(config)
    key = f'{COS_PREFIX}{backup_name}.zip'
    print(f'[3/3] 上传 -> cos://{cfg["COS_BUCKET"]}/{key}')
    try:
        response = client.upload_file(
            Bucket=cfg['COS_BUCKET'],
            Key=key,
            LocalFilePath=local_path,
            EnableMD5=False,
        )
        print(f'      ETag: {response.get("ETag", "?")}')
        return True
    except Exception as e:
        print(f'[ERR] 上传失败: {e}')
        return False


def list_backups(cfg):
    """列出 COS 已有 minimind 备份."""
    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(Region=cfg['COS_REGION'],
                       SecretId=cfg['COS_SECRET_ID'],
                       SecretKey=cfg['COS_SECRET_KEY'])
    client = CosS3Client(config)
    print(f'=== minimind dataset 备份列表 (cos://{cfg["COS_BUCKET"]}/{COS_PREFIX}) ===')
    try:
        resp = client.list_objects(Bucket=cfg['COS_BUCKET'], Prefix=COS_PREFIX)
        contents = resp.get('Contents', [])
        if not contents:
            print('  (空, 暂无 minimind dataset 备份)')
            return []
        backups = []
        for item in contents:
            key = item['Key']
            size = int(item.get('Size', 0))
            mtime = item.get('LastModified', '?')
            backups.append((key, size, mtime))
            print(f'  {key}  {size / 1e6:8.1f}MB  {mtime}')
        return backups
    except Exception as e:
        print(f'[ERR] 列出失败: {e}')
        return []


def main():
    ap = argparse.ArgumentParser(description='dataset 备份到腾讯云 COS')
    ap.add_argument('--list', action='store_true', help='列出已有备份')
    ap.add_argument('--keep-local', action='store_true', help='保留本地压缩包 (默认删除)')
    ap.add_argument('--dry-run', action='store_true', help='只打包不上传 (测试)')
    ap.add_argument('--prefix', default=BACKUP_PREFIX, help='备份名前缀')
    args = ap.parse_args()

    cfg = load_cos_config()

    if args.list:
        list_backups(cfg)
        return

    # 唯一时间戳备份名
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'{args.prefix}_{ts}'
    print(f'=== dataset 备份: {backup_name} ===')

    # 打包
    local_path, src_mb = make_archive(backup_name, dry_run=args.dry_run)

    if args.dry_run:
        print(f'[2/3] [DRY-RUN] 跳过打包 (源数据约 {src_mb:.0f}MB)')
        upload_to_cos(cfg, backup_name, local_path, dry_run=True)
        print('=== DRY-RUN 完成 (未上传) ===')
        return

    # 上传
    ok = upload_to_cos(cfg, backup_name, local_path)
    if not ok:
        sys.exit('[ERR] 备份失败 (上传错误)')

    # 清理本地压缩包 (默认)
    if not args.keep_local:
        os.remove(local_path)
        print(f'      本地压缩包已删除: {local_path}')
    else:
        print(f'      本地压缩包保留: {local_path}')

    print('=== 备份完成 ===')


if __name__ == '__main__':
    main()
