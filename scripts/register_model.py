#!/usr/bin/env python3
"""
模型产物自动登记工具 (AGENTS.md 模型输出规范的执行器)

功能:
  1. 读取模型权重/量化/部署产物的文件大小与时间
  2. 自动更新 docs/MODELS.md (模型清单 + 输出物位置)
  3. 自动追加 CHANGELOG.md 变更说明条目

用法:
  python scripts/register_model.py --name "H2 RAFT v4" \
      --weight out/full_sft_h2_raft_v4_384_ple.pth \
      --ple1 models/full_sft_h2_raft_v4_h384_ple1.bin \
      --int4 models/full_sft_h2_raft_v4_384_int4_g32.pth \
      --deploy ../esp32-ai/firmware/model_v5/H2/model_llm.bin \
      --data "sft_medical_raft (8K, 负样本)" --loss "2.70" --verify "PASS diff 0.00001"

说明:
  - 至少提供 --name 和 --weight; 其余产物可选
  - 文件大小自动从文件系统读取
  - MODELS.md 更新: 在 H1/H2/H3 系列表追加行
  - CHANGELOG.md 更新: 在 Unreleased 区追加条目
"""

import argparse
import datetime
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_MD = os.path.join(ROOT, 'docs', 'MODELS.md')
CHANGELOG_MD = os.path.join(ROOT, 'CHANGELOG.md')


def fmt_size(path):
    if not path or not os.path.exists(path):
        return None
    return os.path.getsize(path)


def mb(bytes_):
    return round(bytes_ / 1e6, 2) if bytes_ else None


def detect_series(name):
    """从模型名推断架构系列 (H1/H2/H3)."""
    m = re.search(r'(H[123])', name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def update_models_md(name, series, weight, ple1, int4, deploy, data, loss, verify):
    if not os.path.exists(MODELS_MD):
        print(f'[skip] MODELS.md 不存在: {MODELS_MD}')
        return
    with open(MODELS_MD, 'r', encoding='utf-8') as f:
        content = f.read()

    # 构建登记行 (不带前导 '-', 由写入时统一加)
    entry = []
    if weight:
        sz = mb(fmt_size(weight))
        entry.append(f'**{name}** | `{weight}` | {sz}MB | {data or "-"} | - | **{loss or "-"}**')
    notes = []
    if ple1:
        sz = mb(fmt_size(ple1))
        notes.append(f'PLE1: `{ple1}` ({sz}MB)')
    if int4:
        sz = mb(fmt_size(int4))
        notes.append(f'int4: `{int4}` ({sz}MB)')
    if deploy:
        sz = mb(fmt_size(deploy))
        notes.append(f'部署: `{deploy}` ({sz}MB)')
    if verify:
        notes.append(f'verify {verify}')
    if notes:
        entry.append('  说明: ' + ' | '.join(notes))

    with open(MODELS_MD, 'w', encoding='utf-8') as f:
        if '## 九、最近模型登记' not in content:
            content += '\n---\n\n## 九、最近模型登记 (自动)\n\n'
        content += '\n'.join(['- ' + e for e in entry]) + '\n'
        f.write(content)
    print(f'[ok] MODELS.md 已更新: {name}')
    return entry


def update_changelog(name, entries):
    if not os.path.exists(CHANGELOG_MD):
        print(f'[skip] CHANGELOG.md 不存在: {CHANGELOG_MD}')
        return
    with open(CHANGELOG_MD, 'r', encoding='utf-8') as f:
        content = f.read()
    # 构建规范条目: 首行主登记 + 说明行 (去掉外层重复 name)
    block = []
    for i, e in enumerate(entries):
        e_clean = e.lstrip('- ').strip()
        if i == 0:
            block.append('- ' + e_clean)  # 主行直接带 name
        else:
            block.append('  - ' + e_clean)
    addition = '\n'.join(block) + '\n\n'
    if '## [Unreleased]' in content:
        idx = content.index('## [Unreleased]') + len('## [Unreleased]')
        content = content[:idx] + '\n' + addition + content[idx:]
    else:
        content = '# Changelog\n\n## [Unreleased]\n\n' + addition + content
    with open(CHANGELOG_MD, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'[ok] CHANGELOG.md 已更新: {name}')


def main():
    ap = argparse.ArgumentParser(description='模型产物自动登记 (MODELS.md + CHANGELOG.md)')
    ap.add_argument('--name', required=True, help='模型/版本名 (如 H2 RAFT v4)')
    ap.add_argument('--weight', help='训练权重路径 (out/*.pth)')
    ap.add_argument('--ple1', help='PLE1 导出路径 (models/*_ple1.bin)')
    ap.add_argument('--int4', help='int4 量化路径 (models/*_int4_g32.pth)')
    ap.add_argument('--deploy', help='部署产物路径 (esp32-ai model_llm.bin)')
    ap.add_argument('--data', help='训练数据说明')
    ap.add_argument('--loss', help='最终 loss')
    ap.add_argument('--verify', help='verify 结果')
    args = ap.parse_args()

    if not args.weight and not args.ple1 and not args.deploy:
        print('[err] 至少提供 --weight / --ple1 / --deploy 之一')
        sys.exit(1)

    series = detect_series(args.name)
    entry = update_models_md(args.name, series, args.weight, args.ple1, args.int4,
                             args.deploy, args.data, args.loss, args.verify)
    if entry:
        changelog_lines = entry
        update_changelog(args.name, changelog_lines)
    print('\n完成。请人工复核 MODELS.md 与 CHANGELOG.md 的格式。')


if __name__ == '__main__':
    main()
