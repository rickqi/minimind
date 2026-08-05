#!/bin/bash
# H3 混合模型 RAFT 微调 v2: 30% 无证据样本 (防遗忘内在知识)
cd /mnt/d/codes/minimind/trainer
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

pkill -f "train_full_sft.py" 2>/dev/null
sleep 2
rm -f /mnt/d/codes/minimind/out/full_sft_h3_mixed_raft.log
rm -f /mnt/d/codes/minimind/out/full_sft_h3_mixed_raft_512_ple.pth
rm -f /mnt/d/codes/minimind/checkpoints/full_sft_h3_mixed_raft_512_ple*.pth

echo "=== starting H3 mixed+RAFT (30% no-evidence) at $(date) ==="
exec python3 -u train_full_sft.py \
    --use_ple 1 --ple_dim 128 \
    --hidden_size 512 --num_hidden_layers 8 \
    --max_seq_len 512 --batch_size 8 --accumulation_steps 1 \
    --epochs 3 --learning_rate 2e-5 \
    --num_workers 4 \
    --data_path ../dataset/sft_medical_raft.jsonl \
    --from_weight full_sft_h3_mixed \
    --save_weight full_sft_h3_mixed_raft --save_dir ../out \
    --save_interval 2000 --log_interval 100 \
    --dtype bfloat16 2>&1 | tee ../out/full_sft_h3_mixed_raft.log

