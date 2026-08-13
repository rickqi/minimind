#!/usr/bin/env bash
# chain_pretrain3ep_sft.sh — B 方案: 当前1epoch跑完 → 续训2epoch(共3) → verify → SFT → verify
# 等待当前 train_pretrain.py 退出, 再续训, 全程无人值守
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"
CKPT="${PROJECT_ROOT}/checkpoints"
PRETRAIN_1EP="${OUT}/email_pretrain_1_256.pth"          # 当前1epoch跑完的权重
PRETRAIN_3EP_NAME="email_pretrain_3ep"                   # 续训2epoch后的3epoch权重名
PRETRAIN_3EP="${OUT}/${PRETRAIN_3EP_NAME}_256.pth"
SFT_DATA="sft_email_train_full.jsonl"
SFT_WEIGHT="${OUT}/email_sft_dense_h256_256.pth"
LOG="${OUT}/chain_pretrain3ep_sft.log"
mkdir -p "${OUT}" "${CKPT}"
echo "[chain-B] 启动 $(date)" | tee "${LOG}"

echo "[chain-B] [1/5] 等待当前 1-epoch 预训练进程退出..." | tee -a "${LOG}"
while pgrep -f "train_pretrain.py" >/dev/null 2>&1; do sleep 30; done
if [ ! -f "${PRETRAIN_1EP}" ]; then
    echo "[chain-B] ❌ 1-epoch 权重未找到 ${PRETRAIN_1EP}, 中止" | tee -a "${LOG}"; exit 1
fi
echo "[chain-B] 1-epoch 完成: ${PRETRAIN_1EP} ($(date))" | tee -a "${LOG}"

echo "[chain-B] [2/5] 续训 2 epoch (从 1-epoch 权重, 共 3 epoch, batch=64 compile)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_pretrain.py \
    --hidden_size 256 --num_hidden_layers 6 --max_seq_len 256 \
    --batch_size 64 --accumulation_steps 4 --epochs 2 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/pretrain_email.jsonl" \
    --from_weight email_pretrain_1 --from_resume 0 \
    --save_weight "${PRETRAIN_3EP_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 100 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
echo "[chain-B] 续训完成: ${PRETRAIN_3EP} ($(date))" | tee -a "${LOG}"

echo "[chain-B] [3/5] verify 3-epoch 预训练权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" \
    --weight "${PRETRAIN_3EP}" --hidden_size 256 --num_hidden_layers 6 --json 2>&1 | tee -a "${LOG}"

echo "[chain-B] [4/5] SFT 从 3-epoch 权重续训 (sft_email_train_full 342K, 2 epoch)..." | tee -a "${LOG}"
bash "${SKILL_DIR}/scripts/train_mode1_default_sft.sh" "${SFT_DATA}" 2 256 6 "${PRETRAIN_3EP_NAME}" 2>&1 | tee -a "${LOG}"

echo "[chain-B] [5/5] verify SFT 权重..." | tee -a "${LOG}"
if [ -f "${SFT_WEIGHT}" ]; then
    python3 "${SKILL_DIR}/scripts/verify_weights.py" \
        --weight "${SFT_WEIGHT}" --hidden_size 256 --num_hidden_layers 6 --json 2>&1 | tee -a "${LOG}"
    echo "[chain-B] ✅✅ 全链路完成 (3ep pretrain → SFT) $(date)" | tee -a "${LOG}"
else
    echo "[chain-B] ⚠️ SFT 权重未找到 ${SFT_WEIGHT}" | tee -a "${LOG}"
fi
