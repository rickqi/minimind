#!/usr/bin/env bash
# train_mode1_default_sft.sh — 手段1: 默认自带模式 (Dense, use_ple=0) SFT 训练
# 用法: bash scripts/train_mode1_default_sft.sh [data_file] [epochs] [hidden_size] [layers] [from_weight]
#   from_weight = none (从零) 或 email_pretrain_1 (预热链: 从 pretrain 权重续训)
set -euo pipefail

DATA_FILE=${1:-sft_email_mixed_400.jsonl}
EPOCHS=${2:-2}
HIDDEN=${3:-256}
LAYERS=${4:-6}
FROM_WEIGHT=${5:-none}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_WEIGHT="email_sft_dense_h${HIDDEN}"
mkdir -p "${PROJECT_ROOT}/out" "${PROJECT_ROOT}/trainer"

# 依赖检查: datasets 缺失时自动安装 (SFTDataset 依赖)
python3 -c "import datasets" 2>/dev/null || pip install -q datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [手段1 默认模式] Dense SFT ==="
echo "  架构: d${HIDDEN}/l${LAYERS} | use_ple=0 | 数据: ${DATA_FILE} | epochs: ${EPOCHS} | from: ${FROM_WEIGHT}"
echo "  输出: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}.pth"

cd "${PROJECT_ROOT}/trainer"

python3 -u train_full_sft.py \
    --use_ple 0 \
    --hidden_size "${HIDDEN}" \
    --num_hidden_layers "${LAYERS}" \
    --max_seq_len 256 \
    --batch_size 8 \
    --accumulation_steps 2 \
    --epochs "${EPOCHS}" \
    --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/${DATA_FILE}" \
    --from_weight "${FROM_WEIGHT}" \
    --save_weight "${SAVE_WEIGHT}" \
    --save_dir "../out" \
    --save_interval 1000 \
    --log_interval 10 \
    --num_workers 2 \
    --dtype bfloat16 \
    2>&1 | tee "../out/${SAVE_WEIGHT}.log"

echo "=== 完成: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}.pth ==="
