#!/bin/bash
# H1 RAFT v4 微调 (从 full_sft_h1 用 sft_medical_raft 8000条, 负样本10%+医学过滤; 含4优化)
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_full_sft.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/full_sft_h1_raft_v4.log
rm -f /mnt/d/codes/minimind/out/full_sft_h1_raft_v4_256_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/full_sft_h1_raft_v4_256_ple*.pth

echo "=== starting H1 RAFT SFT at $(date) ==="
exec python3 -u train_full_sft.py \
    --use_ple 1 --ple_dim 96 \
    --hidden_size 256 --num_hidden_layers 6 \
    --max_seq_len 512 --batch_size 8 --accumulation_steps 1 \
    --epochs 3 --learning_rate 2e-5 \
    --num_workers 4 \
    --data_path ../dataset/sft_medical_raft.jsonl \
    --from_weight full_sft_h1 \
    --save_weight full_sft_h1_raft_v4 --save_dir ../out \
    --save_interval 2000 --log_interval 100 \
    --dtype bfloat16 2>&1 | tee ../out/full_sft_h1_raft_v4.log
