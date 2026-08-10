#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset 恢复脚本: 从腾讯云 COS 下载备份 -> 解压到本地 dataset/

特性:
- 列出 COS 已有 minimind 备份, 选择恢复
- 默认恢复到最新备份; 可指定具体备份名
- 恢复前自动备份当前 dataset/ (防误覆盖)
- 支持 zip 压缩备份恢复

用法:
  python scripts/restore_dataset.py --list                # 列出可用备份
  python scripts/restore_dataset.py                       # 恢复最新备份
  python scripts/restore_dataset.py --backup minimind-dataset-backup_20260810_123456  # 指定备份
  python scripts/restore_dataset.py --dry-run             # 只查不下载
"""
import argparse
import datetime
import os
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
ENV_FILE = os.path.join(PROJECT_ROOT, '.env.cosine')
# 前缀带 minimind, 与备份脚本一致, 存 backups/minimind/ 子目录
BACKUP_PREFIX = 'minimind-dataset-backup'
COS_PREFIX = 'backups/minimind/'


def load_cos_config():
    """从 .env.cosine 加载 COS 配置."""
    if not os.path.exists(ENV_FILE):
        sys.exit(f'[ERR] 未找到密钥文件 {ENV_FILE}')
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


def get_client(cfg):
    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(Region=cfg['COS_REGION'],
                       SecretId=cfg['COS_SECRET_ID'],
                       SecretKey=cfg['COS_SECRET_KEY'])
    return CosS3Client(config)


def list_backups(cfg):
    """列出 COS 备份, 返回 [(key, size, mtime)]."""
    client = get_client(cfg)
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


def pre_backup_current():
    """恢复前备份当前 dataset/ 为本地 zip (防误覆盖)."""
    if not os.path.isdir(DATASET_DIR):
        return None
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bak_path = os.path.join(PROJECT_ROOT, f'pre_restore_{ts}.zip')
    print(f'[安全] 备份当前 dataset/ -> {bak_path}')
    with zipfile.ZipFile(bak_path, 'w', zipfile.ZIP_BZIP2, compresslevel=9) as zf:
        for f in sorted(os.listdir(DATASET_DIR)):
            p = os.path.join(DATASET_DIR, f)
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.join('dataset', f))
    return bak_path


def download_and_restore(cfg, backup_key, dry_run=False):
    """下载备份并解压到 dataset/."""
    client = get_client(cfg)
    backup_name = os.path.basename(backup_key)  # minimind-dataset-backup_*.zip
    local_zip = os.path.join(PROJECT_ROOT, backup_name)

    if not dry_run:
        print(f'[1/3] 下载 {backup_key}')
        client.download_file(
            Bucket=cfg['COS_BUCKET'],
            Key=backup_key,
            DestFilePath=local_zip,
        )
        print(f'      完成: {os.path.getsize(local_zip) / 1e6:.1f}MB')

    print(f'[2/3] 解压到 {DATASET_DIR}')
    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    if dry_run:
        print('[3/3] [DRY-RUN] 跳过下载/解压/删除')
        return

    # 清空当前 dataset/ 数据文件 (保留非数据文件如 __init__.py/lm_dataset.py)
    # 注意: 必须在 dry_run 判断之后, 避免 dry-run 误删
    for f in os.listdir(DATASET_DIR):
        if f.endswith('.jsonl') or f.endswith('.bin') or f.endswith('.csv'):
            p = os.path.join(DATASET_DIR, f)
            if os.path.isfile(p):
                os.remove(p)
                print(f'      移除旧数据: {f}')

    with zipfile.ZipFile(local_zip, 'r') as zf:
        zf.extractall(PROJECT_ROOT)
    os.remove(local_zip)
    print(f'      解压完成, 压缩包已清理')

    # 列出恢复结果
    print('[3/3] 恢复结果:')
    total = 0
    for f in sorted(os.listdir(DATASET_DIR)):
        p = os.path.join(DATASET_DIR, f)
        if os.path.isfile(p) and f.endswith('.jsonl'):
            sz = os.path.getsize(p)
            total += sz
            print(f'  {sz / 1e6:9.1f}MB  {f}')
    print(f'  总计: {total / 1e6:.0f}MB')


def main():
    ap = argparse.ArgumentParser(description='从腾讯云 COS 恢复 dataset')
    ap.add_argument('--list', action='store_true', help='列出可用备份')
    ap.add_argument('--backup', default=None, help='指定备份名 (默认最新)')
    ap.add_argument('--dry-run', action='store_true', help='只查不下载')
    args = ap.parse_args()

    cfg = load_cos_config()
    backups = list_backups(cfg)
    if not backups:
        print('[ERR] 无可用备份')
        return

    if args.list:
        return

    # 选择备份: 指定 或 最新
    if args.backup:
        target = [b for b in backups if args.backup in b[0]]
        if not target:
            sys.exit(f'[ERR] 未找到备份含 "{args.backup}"')
        backup_key, size, mtime = target[0]
    else:
        backup_key, size, mtime = backups[-1]  # list_objects 按字典序, 时间戳最新在后
    print(f'=== 恢复备份: {backup_key} ({size / 1e6:.1f}MB, {mtime}) ===')

    # 安全: 备份当前数据
    if not args.dry_run:
        pre_backup_current()

    download_and_restore(cfg, backup_key, dry_run=args.dry_run)
    print('=== 恢复完成 ===')


if __name__ == '__main__':
    main()
