#!/bin/bash
cd /mnt/d/codes/minimind
export PYTHONIOENCODING=utf-8
python3 -u - <<'PYEOF'
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scripts')
from rag_medical import cmd_chat

questions = ['高血压的诊断标准是什么', '糖尿病的临床表现有哪些', '感染性休克的血象检查有什么特点']
for q in questions:
    print('\n' + '='*60)
    print('Q:', q)
    cmd_chat(q, 384, 8, 128, 'full_sft_h2_raft')
PYEOF
