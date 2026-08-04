#!/bin/bash
# 用 DPO 权重导出 3 个模型的 int4 量化 + PLE1 二进制
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8

for spec in "h1 256 6 96" "h2 384 8 128" "h3 512 8 128"; do
    set -- $spec
    tag=$1; dim=$2; layers=$3; p=$4
    echo "======== dpo_${tag} (d${dim}/l${layers}/p${p}) ========"
    echo "--- int4 quant ---"
    python3 -u scripts/quantize_ple.py \
        --hidden_size $dim --num_hidden_layers $layers --ple_dim $p \
        --weight dpo_${tag} --group 32 --device cuda 2>&1 | grep -v 'Generating train split' | tail -4
    echo "--- ple1 export ---"
    python3 -u scripts/export_ple1.py \
        --hidden_size $dim --num_hidden_layers $layers --ple_dim $p \
        --num_attention_heads 8 --num_key_value_heads 4 \
        --weight dpo_${tag} 2>&1 | tail -4
    echo
done
echo "=== models/ contents ==="
ls -la models/ | grep -E 'dpo'
