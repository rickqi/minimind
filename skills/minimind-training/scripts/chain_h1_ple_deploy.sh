#!/usr/bin/env bash
# chain_h1_ple_deploy.sh — H1 PLE 部署链: 等 SFT 权重 → int4 量化 → PLE1 导出 + golden
# 体现 PLE 相对 Dense 的核心价值 (int4 量化 + PLE1 扁平格式, 适配 ESP32 flash 驻留)
# 注: esp32-ai 不在本项目, convert_h2/verify_h2 实机步骤跳过 (本项目止于主机产物)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"
MODELS="${PROJECT_ROOT}/models"
SFT_NAME="email_sft_h1ple"
SFT="${OUT}/${SFT_NAME}_256_ple.pth"
LOG="${OUT}/chain_h1_ple_deploy.log"
DATA="${SKILL_DIR}/dataset/pretrain_email.jsonl"
mkdir -p "${OUT}" "${MODELS}"
echo "[H1PLE-Deploy] 启动 $(date)" | tee "${LOG}"

echo "[H1PLE-Deploy] 等待 SFT 权重产出: ${SFT}" | tee -a "${LOG}"
while [ ! -f "${SFT}" ]; do sleep 60; done
sleep 30  # 等 SFT 进程写完落盘
echo "[H1PLE-Deploy] SFT 权重就绪 ($(date))" | tee -a "${LOG}"

echo "[H1PLE-Deploy] [1/2] int4 量化 (group=32, 量化 deg 报告)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
python3 scripts/quantize_ple.py \
    --weight "${SFT_NAME}" --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 \
    --save_dir "${OUT}" --export_dir "${MODELS}" --group 32 --bits 4 \
    --data_path "${DATA}" --max_seq_len 128 --val_iters 10 --device cuda 2>&1 | tee -a "${LOG}"
echo "[H1PLE-Deploy] int4 产物: ${MODELS}/${SFT_NAME}_256_int4_g32.pth ($(date))" | tee -a "${LOG}"

echo "[H1PLE-Deploy] [2/2] PLE1 扁平导出 + golden..." | tee -a "${LOG}"
python3 scripts/export_ple1.py \
    --weight "${SFT_NAME}" --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 \
    --num_attention_heads 8 --num_key_value_heads 4 --seq_len 256 \
    --save_dir "${OUT}" --out_dir "${MODELS}" --group 32 --bits 4 2>&1 | tee -a "${LOG}"
PLE1="${MODELS}/${SFT_NAME}_h256_ple1.bin"
GOLDEN="${MODELS}/${SFT_NAME}_h256_golden.npz"
echo "[H1PLE-Deploy] PLE1: ${PLE1} ($(stat -c%s "${PLE1}" 2>/dev/null || echo '?') bytes)" | tee -a "${LOG}"
echo "[H1PLE-Deploy] golden: ${GOLDEN}" | tee -a "${LOG}"
echo "[H1PLE-Deploy] ✅✅ 部署产物导出完成 $(date)" | tee -a "${LOG}"
echo "[H1PLE-Deploy] 后续(esp32-ai 仓库): convert_h2.py → firmware/model_v5/H1/model_llm.bin → verify_h2.c PASS" | tee -a "${LOG}"
