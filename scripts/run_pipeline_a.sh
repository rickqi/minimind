#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
rm -f dataset/pretrain_medical.jsonl out/medical_pretrain_report.json
echo "=== run pipeline A (3 sources) ==="
python3 -u scripts/build_medical_pretrain.py 2>&1 | tail -20

