#!/usr/bin/env bash
# chain_sft_after_pretrain.sh — 预训练完成后自动接 verify → SFT → verify
# 等待 out/email_pretrain_1_256.pth 产出且预训练进程退出, 再执行后续链路
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"
PRETRAIN_WEIGHT="${OUT}/email_pretrain_1_256.pth"
SFT_DATA="sft_email_train_full.jsonl"
SFT_WEIGHT_NAME="email_sft_dense_h256"
SFT_WEIGHT="${OUT}/${SFT_WEIGHT_NAME}_256.pth"
CHAIN_LOG="${OUT}/chain_sft.log"
echo "[chain] 启动链式管道 $(date)" | tee "${CHAIN_LOG}"

echo "[chain] [1/4] 等待预训练权重产出 + 进程退出..." | tee -a "${CHAIN_LOG}"
while true; do
    if [ -f "${PRETRAIN_WEIGHT}" ] && ! pgrep -f "train_pretrain.py" >/dev/null 2>&1; then
        break
    fi
    sleep 30
done
echo "[chain] 预训练完成: ${PRETRAIN_WEIGHT} ($(date))" | tee -a "${CHAIN_LOG}"

echo "[chain] [2/4] verify 预训练权重 (missing=0/unexpected=0)..." | tee -a "${CHAIN_LOG}"
cd "${PROJECT_ROOT}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" \
    --weight "${PRETRAIN_WEIGHT}" --hidden_size 256 --num_hidden_layers 6 --json 2>&1 | tee -a "${CHAIN_LOG}"
vp_rc=${PIPESTATUS[0]}
if [ "${vp_rc}" -ne 0 ]; then
    echo "[chain] ❌ 预训练权重校验失败 (rc=${vp_rc}), 中止" | tee -a "${CHAIN_LOG}"
    exit 1
fi
echo "[chain] ✅ 预训练权重校验通过" | tee -a "${CHAIN_LOG}"

echo "[chain] [3/4] SFT 从预热权重续训 (sft_email_train_full, 342K, 2 epoch)..." | tee -a "${CHAIN_LOG}"
bash "${SKILL_DIR}/scripts/train_mode1_default_sft.sh" "${SFT_DATA}" 2 256 6 email_pretrain_1 2>&1 | tee -a "${CHAIN_LOG}"
echo "[chain] SFT 完成: ${SFT_WEIGHT} ($(date))" | tee -a "${CHAIN_LOG}"

echo "[chain] [4/4] verify SFT 权重..." | tee -a "${CHAIN_LOG}"
if [ -f "${SFT_WEIGHT}" ]; then
    python3 "${SKILL_DIR}/scripts/verify_weights.py" \
        --weight "${SFT_WEIGHT}" --hidden_size 256 --num_hidden_layers 6 --json 2>&1 | tee -a "${CHAIN_LOG}"
    echo "[chain] ✅✅ 全链路完成 $(date)" | tee -a "${CHAIN_LOG}"
else
    echo "[chain] ⚠️ SFT 权重未找到 ${SFT_WEIGHT}, 检查 SFT 日志" | tee -a "${CHAIN_LOG}"
fi
