#!/usr/bin/env bash
# chain_h1_ple_pipeline.sh — H1 PLE 手段2 (d256/l6/ple96) 全管线: 从零 pretrain 3ep → verify → SFT → verify → eval
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"
PRE="${OUT}/email_pretrain_h1ple_256_ple.pth"
PRE_NAME="email_pretrain_h1ple"
SFT="${OUT}/email_sft_h1ple_256_ple.pth"
SFT_NAME="email_sft_h1ple"
LOG="${OUT}/chain_h1_ple_pipeline.log"
echo "[H1PLE] 启动 $(date)" | tee "${LOG}"

echo "[H1PLE] [1/5] 从零预训练 3 epoch (d256/l6/ple96, batch=32 compile, 385 samp/s)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_pretrain.py \
    --use_ple 1 --ple_dim 96 --hidden_size 256 --num_hidden_layers 6 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 3 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/pretrain_email.jsonl" \
    --from_weight none --from_resume 1 \
    --save_weight "${PRE_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
echo "[H1PLE] 预训练完成: ${PRE} ($(date))" | tee -a "${LOG}"

echo "[H1PLE] [2/5] verify 预训练权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${PRE}" --hidden_size 256 --num_hidden_layers 6 --use_ple --ple_dim 96 --json 2>&1 | tee -a "${LOG}"

echo "[H1PLE] [3/5] SFT (sft_email_train_full 342K, 2ep, from PLE预训练)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_full_sft.py \
    --use_ple 1 --ple_dim 96 --hidden_size 256 --num_hidden_layers 6 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 2 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/sft_email_train_full.jsonl" \
    --from_weight "${PRE_NAME}" --from_resume 1 \
    --save_weight "${SFT_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"
echo "[H1PLE] SFT 完成: ${SFT} ($(date))" | tee -a "${LOG}"

echo "[H1PLE] [4/5] verify SFT 权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${SFT}" --hidden_size 256 --num_hidden_layers 6 --use_ple --ple_dim 96 --json 2>&1 | tee -a "${LOG}"

echo "[H1PLE] [5/5] eval SFT 权重..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/eval_email.py" --weight "${SFT}" --hidden_size 256 --num_hidden_layers 6 --use_ple --ple_dim 96 --per_type 2 --device cuda 2>&1 | tee -a "${LOG}"
echo "[H1PLE] ✅✅ 全管线完成 $(date)" | tee -a "${LOG}"
