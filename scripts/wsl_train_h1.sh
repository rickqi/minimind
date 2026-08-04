#!/bin/bash
# 训练启动脚本 - 由 PowerShell Start-Process 保持 wsl 会话存活
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_pretrain.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/pretrain_h1.log
rm -f /mnt/d/codes/minimind/out/pretrain_h1_256_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/pretrain_h1_256_ple*.pth

echo "=== starting training at $(date) ==="
exec python3 -u train_pretrain.py \
    --use_ple 1 --ple_dim 96 \
    --hidden_size 256 --num_hidden_layers 6 \
    --max_seq_len 128 --batch_size 16 --accumulation_steps 1 \
    --epochs 1 --learning_rate 1e-3 \
    --num_workers 4 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --save_weight pretrain_h1 --save_dir ../out \
    --save_interval 5000 --log_interval 100 \
    --dtype bfloat16 2>&1 | tee ../out/pretrain_h1.log
