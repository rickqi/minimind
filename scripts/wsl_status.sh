#!/bin/bash
echo "=== log file ==="
ls -la /mnt/d/codes/minimind/out/pretrain_h1.log 2>&1
echo "=== gpu compute apps ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
echo "=== python procs ==="
ps -ef | grep -i python | grep -v grep | head -10
echo "=== log mtime ==="
stat -c '%y %s' /mnt/d/codes/minimind/out/pretrain_h1.log 2>&1
