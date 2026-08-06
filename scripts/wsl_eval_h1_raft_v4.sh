#!/bin/bash
# H1 RAFT v4 问答验证 (负样本/盲引修复) — 基于 wsl_eval_raft_v4.sh 适配 H1
# 实现: scripts/eval_h1_raft_v4.py (独立文件, 避免 heredoc 转义问题)
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u scripts/eval_h1_raft_v4.py
