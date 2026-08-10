#!/usr/bin/env bash
# train_pretrain.sh — 预训练预热 (手段1/2 共用, 用 --mode 区分架构)
# 用法: bash scripts/train_pretrain.sh [mode] [data_file] [epochs] [hidden_size] [layers] [ple_dim]
#   mode 1 = Dense (use_ple=0), mode 2 = PLE (use_ple=1)
set -euo pipefail

MODE=${1:-1}
DATA_FILE=${2:-pretrain_email.jsonl}
EPOCHS=${3:-3}
HIDDEN=${4:-256}
LAYERS=${5:-6}
PLE_DIM=${6:-96}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_WEIGHT="email_pretrain_${MODE}"
mkdir -p "${PROJECT_ROOT}/out" "${PROJECT_ROOT}/trainer"

# 依赖检查: datasets 缺失时自动安装
python3 -c "import datasets" 2>/dev/null || pip install -q datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

PLE_ARG=""
if [ "${MODE}" = "2" ]; then
  PLE_ARG="--use_ple 1 --ple_dim ${PLE_DIM}"
fi

echo "=== 预训练预热 (mode ${MODE}) ==="
echo "  架构: d${HIDDEN}/l${LAYERS}${MODE:+, p${PLE_DIM}} | 数据: ${DATA_FILE} | epochs: ${EPOCHS}"
echo "  输出: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}${MODE:+,_ple}.pth"

cd "${PROJECT_ROOT}/trainer"

python3 -u train_pretrain.py \
    ${PLE_ARG} \
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

echo "=== 完成: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}.pth ==="
