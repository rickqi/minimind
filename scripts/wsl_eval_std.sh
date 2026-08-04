#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
# 自动测试模式: 输入 "0" 走内置自动测试(不依赖手动输入)
(echo "0"; sleep 2) | python3 -u eval_llm.py \
    --load_from ./model --weight full_sft_h1 \
    --hidden_size 256 --num_hidden_layers 6 \
    --use_ple 1 --ple_dim 96 \
    --max_new_tokens 80 --temperature 0.85 --top_p 0.85 2>&1 | tail -20
