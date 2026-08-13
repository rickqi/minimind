#!/usr/bin/env bash
# train_pretrain_auto.sh — 训练前自动评估硬件 + 调参, 再启动预训练
# 用法: bash train_pretrain_auto.sh [mode] [data_file] [epochs] [hidden] [layers] [ple_dim]
#   mode 1=Dense(use_ple=0) 2=PLE(use_ple=1)
# 流程: 抽样 16K → hardware_profile.py 测吞吐拐点 → 用推荐 batch/workers + use_compile + from_resume 启动
set -euo pipefail

MODE=${1:-1}
DATA_FILE=${2:-pretrain_email.jsonl}
EPOCHS=${3:-1}
HIDDEN=${4:-256}
LAYERS=${5:-6}
PLE_DIM=${6:-96}
MAX_SEQ_LEN=${7:-256}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_WEIGHT="email_pretrain_${MODE}"
DATA_PATH="${SKILL_DIR}/dataset/${DATA_FILE}"
mkdir -p "${PROJECT_ROOT}/out"

python3 -c "import datasets" 2>/dev/null || pip install -q datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "============================================================"
echo "[1/3] 硬件探测 + 参数调优"
echo "============================================================"
PROBE=/tmp/_pretrain_probe_$$.jsonl
head -n 16000 "${DATA_PATH}" > "${PROBE}"
trap 'rm -f "${PROBE}"' EXIT

HW_JSON=$(python3 "${SCRIPTS_DIR}/hardware_profile.py" \
    --data_path "${PROBE}" --hidden "${HIDDEN}" --layers "${LAYERS}" \
    --max_seq_len "${MAX_SEQ_LEN}" --device cuda --dtype bfloat16 \
    --max_batch 256 --bench_steps 12 2>/dev/null)

BATCH=$(echo "${HW_JSON}" | python3 -c "import sys,json;print(json.load(sys.stdin)['batch_size'])")
WORKERS=$(echo "${HW_JSON}" | python3 -c "import sys,json;print(json.load(sys.stdin)['num_workers'])")
SPS=$(echo "${HW_JSON}" | python3 -c "import sys,json;print(json.load(sys.stdin).get('samples_per_s',0))")

# 总样本数 → 单 epoch 预估耗时
TOTAL=$(wc -l < "${DATA_PATH}")
ETA_MIN=$(python3 -c "print(round(${TOTAL}/${SPS}/60,1))" 2>/dev/null || echo "?")

HW_SUMMARY=$(echo "${HW_JSON}" | python3 -c "import sys,json;d=json.load(sys.stdin);p=d['profile'];print(f\"{p['gpu_name']} cap{p['compute_cap']} ram{p['ram_available_gb']}GB cores{p['cpu_cores']}\")")
echo "  硬件: ${HW_SUMMARY}"
echo "  推荐: batch=${BATCH} workers=${WORKERS} seq=${MAX_SEQ_LEN} dtype=bfloat16 + torch.compile"
echo "  吞吐: ${SPS} samp/s | 全量 ${TOTAL} 行 → 单 epoch ≈ ${ETA_MIN} min"

PLE_ARG=""
if [ "${MODE}" = "2" ]; then
    PLE_ARG="--use_ple 1 --ple_dim ${PLE_DIM}"
fi

echo ""
echo "============================================================"
echo "[2/3] 启动预训练 (mode ${MODE}, use_compile=1, from_resume=1)"
echo "  ${PLE_ARG:-Dense} d${HIDDEN}/l${LAYERS} | ${EPOCHS} epoch"
echo "  输出: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}.pth"
echo "============================================================"
cd "${PROJECT_ROOT}/trainer"

python3 -u train_pretrain.py \
    ${PLE_ARG} \
    --hidden_size "${HIDDEN}" \
    --num_hidden_layers "${LAYERS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --batch_size "${BATCH}" \
    --accumulation_steps 4 \
    --epochs "${EPOCHS}" \
    --learning_rate 5e-4 \
    --data_path "${DATA_PATH}" \
    --from_weight none \
    --from_resume 1 \
    --save_weight "${SAVE_WEIGHT}" \
    --save_dir "../out" \
    --save_interval 2000 \
    --log_interval 100 \
    --num_workers "${WORKERS}" \
    --dtype bfloat16 \
    --use_compile 1 \
    2>&1 | tee "../out/${SAVE_WEIGHT}.log"

echo "============================================================"
echo "[3/3] 完成: ${PROJECT_ROOT}/out/${SAVE_WEIGHT}_${HIDDEN}.pth"
echo "============================================================"
