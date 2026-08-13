#!/usr/bin/env bash
# chain_h2_pipeline.sh — H2 (d384/l8 Dense) 全管线: pretrain 3ep → verify → SFT → verify → eval
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"
PRE="${OUT}/email_pretrain_h2_384.pth"
PRE_NAME="email_pretrain_h2"
SFT="${OUT}/email_sft_h2_384.pth"
SFT_NAME="email_sft_h2"
LOG="${OUT}/chain_h2_pipeline.log"
echo "[H2] 启动 $(date)" | tee "${LOG}"

echo "[H2] [1/5] 预训练 3 epoch (d384/l8, batch=32 compile) ETA~7.8h..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_pretrain.py \
    --hidden_size 384 --num_hidden_layers 8 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 3 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/pretrain_email.jsonl" \
    --from_weight none --from_resume 1 \
    --save_weight "${PRE_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
echo "[H2] 预训练完成: ${PRE} ($(date))" | tee -a "${LOG}"

echo "[H2] [2/5] verify 预训练权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${PRE}" --hidden_size 384 --num_hidden_layers 8 --json 2>&1 | tee -a "${LOG}"

echo "[H2] [3/5] SFT (sft_email_train_full 342K, 2ep, from pretrain)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_full_sft.py \
    --use_ple 0 --hidden_size 384 --num_hidden_layers 8 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 2 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/sft_email_train_full.jsonl" \
    --from_weight "${PRE_NAME}" --from_resume 1 \
    --save_weight "${SFT_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
echo "[H2] SFT 完成: ${SFT} ($(date))" | tee -a "${LOG}"

echo "[H2] [4/5] verify SFT 权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${SFT}" --hidden_size 384 --num_hidden_layers 8 --json 2>&1 | tee -a "${LOG}"

echo "[H2] [5/5] eval SFT 权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/eval_email.py" --weight "${SFT}" --hidden_size 384 --num_hidden_layers 8 --per_type 2 --device cuda 2>&1 | tee -a "${LOG}"
echo "[H2] ✅✅ 全管线完成 $(date)" | tee -a "${LOG}"
