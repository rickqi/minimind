#!/usr/bin/env bash
# train_mode2_ple_sft.sh — 手段2: 自有数据分支 (PLE, use_ple=1) SFT 训练
# 用法: bash scripts/train_mode2_ple_sft.sh [data_file] [epochs] [hidden_size] [layers] [ple_dim]
set -euo pipefail

DATA_FILE=${1:-sft_email_mixed_400.jsonl}
EPOCHS=${2:-2}
HIDDEN=${3:-256}
LAYERS=${4:-6}
PLE_DIM=${5:-96}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_WEIGHT="email_sft_ple_h${HIDDEN}"
mkdir -p "${PROJECT_ROOT}/out" "${PROJECT_ROOT}/trainer"

# 依赖检查: datasets 缺失时自动安装 (SFTDataset 依赖)
python3 -c "import datasets" 2>/dev/null || pip install -q datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [手段2 自有数据分支] PLE SFT ==="
echo "  架构: d${HIDDEN}/l${LAYERS}/p${PLE_DIM} | use_ple=1 | 数据: ${DATA_FILE} | epochs: ${EPOCHS}"
echo "  输出: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}_ple.pth"

cd "${PROJECT_ROOT}/trainer"

python3 -u train_full_sft.py \
    --use_ple 1 \
    --ple_dim "${PLE_DIM}" \
    --hidden_size "${HIDDEN}" \
    --num_hidden_layers "${LAYERS}" \
    --max_seq_len 256 \
    --batch_size 8 \
    --accumulation_steps 2 \
    --epochs "${EPOCHS}" \
    --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/${DATA_FILE}" \
    --from_weight none \
    --save_weight "${SAVE_WEIGHT}" \
    --save_dir "../out" \
    --save_interval 1000 \
    --log_interval 10 \
    --num_workers 2 \
    --dtype bfloat16 \
    2>&1 | tee "../out/${SAVE_WEIGHT}.log"

echo "=== 完成: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}_ple.pth ==="
