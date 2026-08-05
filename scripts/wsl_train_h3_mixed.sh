#!/bin/bash
# H3 从零预训练混合数据 (医疗1:2通用): pretrain_mixed.jsonl (386,976条)
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_pretrain.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/pretrain_h3_mixed.log
rm -f /mnt/d/codes/minimind/out/pretrain_h3_mixed_512_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/pretrain_h3_mixed_512_ple*.pth

echo "=== starting H3 mixed-data pretrain at $(date) ==="
exec python3 -u train_pretrain.py \
    --use_ple 1 --ple_dim 128 \
    --hidden_size 512 --num_hidden_layers 8 \
    --max_seq_len 128 --batch_size 16 --accumulation_steps 1 \
    --epochs 1 --learning_rate 1e-3 \
    --num_workers 4 \
    --data_path ../dataset/pretrain_mixed.jsonl \
    --save_weight pretrain_h3_mixed --save_dir ../out \
    --save_interval 5000 --log_interval 100 \
    --dtype bfloat16 2>&1 | tee ../out/pretrain_h3_mixed.log
