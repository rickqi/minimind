#!/bin/bash
# 医疗增强 SFT 续训: 从 full_sft_h2 用 sft_medical_mixed.jsonl 微调 (WSL, RTX 5080)
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_full_sft.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/full_sft_h2_med.log
rm -f /mnt/d/codes/minimind/out/full_sft_h2_med_384_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/full_sft_h2_med_384_ple*.pth

echo "=== starting H2 medical SFT at $(date) ==="
exec python3 -u train_full_sft.py \
    --use_ple 1 --ple_dim 128 \
    --hidden_size 384 --num_hidden_layers 8 \
    --max_seq_len 512 --batch_size 8 --accumulation_steps 1 \
    --epochs 1 --learning_rate 2e-5 \
    --num_workers 4 \
    --data_path ../dataset/sft_medical_mixed.jsonl \
    --from_weight full_sft_h2 \
    --save_weight full_sft_h2_med --save_dir ../out \
    --save_interval 5000 --log_interval 100 \
    --dtype bfloat16 2>&1 | tee ../out/full_sft_h2_med.log
