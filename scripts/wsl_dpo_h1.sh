#!/bin/bash
# H1 级 PLE 模型 DPO 偏好优化 (WSL, RTX 5080)
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_dpo.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/dpo_h1.log
rm -f /mnt/d/codes/minimind/out/dpo_h1_256_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/dpo_h1_256_ple*.pth

echo "=== starting H1 DPO at $(date) ==="
exec python3 -u train_dpo.py \
    --use_ple 1 --ple_dim 96 \
    --hidden_size 256 --num_hidden_layers 6 \
    --max_seq_len 512 --batch_size 4 --accumulation_steps 2 \
    --epochs 1 --learning_rate 4e-8 \
    --num_workers 4 \
    --data_path ../dataset/dpo.jsonl \
    --from_weight full_sft_h1 \
    --save_weight dpo_h1 --save_dir ../out \
    --save_interval 500 --log_interval 50 \
    --beta 0.15 --dtype bfloat16 2>&1 | tee ../out/dpo_h1.log
