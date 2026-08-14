#!/usr/bin/env bash
# chain_h2_ple_pipeline.sh — H2 PLE 手段2 全管线 (串行单脚本, 训练+部署一体)
# d384/l8/ple128, 24.95M, batch=32: 从零 pretrain 3ep → verify → SFT → verify → eval → int4 → PLE1
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_ROOT}/out"; MODELS="${PROJECT_ROOT}/models"
PRE_NAME="email_pretrain_h2ple"; SFT_NAME="email_sft_h2ple"
PRE="${OUT}/${PRE_NAME}_384_ple.pth"; SFT="${OUT}/${SFT_NAME}_384_ple.pth"
LOG="${OUT}/chain_h2_ple_pipeline.log"
DATA="${SKILL_DIR}/dataset/pretrain_email.jsonl"
mkdir -p "${OUT}" "${MODELS}"
echo "[H2PLE] 启动 $(date)" | tee "${LOG}"

echo "[H2PLE] [1/7] 从零预训练 3ep (d384/l8/ple128, batch=32 compile, 231 samp/s, ETA~6h)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_pretrain.py \
    --use_ple 1 --ple_dim 128 --hidden_size 384 --num_hidden_layers 8 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 3 --learning_rate 5e-4 \
    --data_path "${DATA}" --from_weight none --from_resume 1 \
    --save_weight "${PRE_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"; echo "[H2PLE] 预训练完成: ${PRE} ($(date))" | tee -a "${LOG}"

echo "[H2PLE] [2/7] verify 预训练..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${PRE}" --hidden_size 384 --num_hidden_layers 8 --use_ple --ple_dim 128 --json 2>&1 | tee -a "${LOG}"

echo "[H2PLE] [3/7] SFT (sft_email_train_full 342K, 2ep)..." | tee -a "${LOG}"
cd "${PROJECT_ROOT}/trainer"
python3 -u train_full_sft.py \
    --use_ple 1 --ple_dim 128 --hidden_size 384 --num_hidden_layers 8 --max_seq_len 256 \
    --batch_size 32 --accumulation_steps 4 --epochs 2 --learning_rate 5e-4 \
    --data_path "${SKILL_DIR}/dataset/sft_email_train_full.jsonl" \
    --from_weight "${PRE_NAME}" --from_resume 1 \
    --save_weight "${SFT_NAME}" --save_dir "${OUT}" \
    --save_interval 2000 --log_interval 200 --num_workers 6 \
    --dtype bfloat16 --use_compile 1 2>&1 | tee -a "${LOG}"
cd "${PROJECT_ROOT}"; echo "[H2PLE] SFT 完成: ${SFT} ($(date))" | tee -a "${LOG}"

echo "[H2PLE] [4/7] verify SFT..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/verify_weights.py" --weight "${SFT}" --hidden_size 384 --num_hidden_layers 8 --use_ple --ple_dim 128 --json 2>&1 | tee -a "${LOG}"

echo "[H2PLE] [5/7] eval SFT..." | tee -a "${LOG}"
python3 "${SKILL_DIR}/scripts/eval_email.py" --weight "${SFT}" --hidden_size 384 --num_hidden_layers 8 --use_ple --ple_dim 128 --per_type 2 --device cuda 2>&1 | tee -a "${LOG}"

echo "[H2PLE] [6/7] int4 量化 (group=32)..." | tee -a "${LOG}"
python3 scripts/quantize_ple.py --weight "${SFT_NAME}" --hidden_size 384 --num_hidden_layers 8 --ple_dim 128 \
    --save_dir "${OUT}" --export_dir "${MODELS}" --group 32 --bits 4 \
    --data_path "${DATA}" --max_seq_len 128 --val_iters 10 --device cuda 2>&1 | tee -a "${LOG}"

echo "[H2PLE] [7/7] PLE1 导出 + golden..." | tee -a "${LOG}"
python3 scripts/export_ple1.py --weight "${SFT_NAME}" --hidden_size 384 --num_hidden_layers 8 --ple_dim 128 \
    --num_attention_heads 8 --num_key_value_heads 4 --seq_len 256 \
    --save_dir "${OUT}" --out_dir "${MODELS}" --group 32 --bits 4 2>&1 | tee -a "${LOG}"
echo "[H2PLE] ✅✅ 全管线完成 (训练+部署) $(date)" | tee -a "${LOG}"
