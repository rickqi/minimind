---
name: minimind-training
description: "MiniMind 模型训练技能。涵盖两种训练手段 (默认自带 Dense 模式 / 自有数据 PLE 分支模式) 的完整流程: 数据准备、SFT/pretrain/DPO/RAFT 训练、评估、质量检查、产物登记。当用户需要训练 MiniMind 模型、评估训练产物、检查训练质量、或使用 EmailAgent 等自定义数据训练时使用此技能。"
license: Proprietary
---

# MiniMind 训练技能

## Overview

本技能封装 MiniMind 项目的两种训练手段端到端流程。项目定位: 处理训练数据 → 训练模型 → 发布导出 (见项目 AGENTS.md 两类训练模式):

| 手段 | 说明 | 架构开关 | 权重后缀 |
|---|---|---|---|
| **手段1 默认自带模式** | 上游原版训练链路 (pretrain/SFT/DPO), 是基础 | `--use_ple 0` | 无 (dense) |
| **手段2 自有数据分支** | PLE 架构 + RAFT + RAG + 量化部署, 项目扩展 | `--use_ple 1 --ple_dim N` | `_ple` |

两种手段共用同一套训练器 (`trainer/train_*.py`), 仅靠 `--use_ple`/`--ple_dim` 区分。**规则**: 默认模式代码必须保留; PLE 是 `use_ple` 隔离参数, 默认 False 保持上游行为。

## When to Use

- 用户要求"训练模型"、"微调"、"SFT/pretrain/DPO/RAFT"、"用 XX 数据训练"
- 用户要求"评估模型"、"检查训练质量"、"验证权重"
- 使用 EmailAgent 或其他自定义 jsonl 数据训练 MiniMind

## 架构常量 (贯穿全流程)

| 架构 | hidden_size | layers | ple_dim | q_heads | kv_heads | 备注 |
|---|---|---|---|---|---|---|
| **H1** | 256 | 6 | 96 | 8 | 4 | 最小, 验证首选 |
| **H2** | 384 | 8 | 128 | 8 | 4 | 可烧 ESP32 |
| **H3** | 512 | 8 | 128 | 8 | 4 | 仅 PC (超 flash) |

词表 6400 (BPE)。训练环境: WSL + ROCm/CUDA, `torch.cuda.is_available()`。

## 目录结构

```
skills/minimind-training/
├── SKILL.md                    # 本文件
├── dataset/                    # 预处理后的训练数据 (prepare_email_data.py 输出)
│   ├── sft_email_tasks.jsonl   #   纯 conversations (3681 条)
│   ├── sft_email_threads.jsonl #   多轮对话 (137 条)
│   ├── sft_email_mixed.jsonl   #   合并去重 (3432 条, 主训练集)
│   ├── dpo_email.jsonl         #   DPO 偏好对 (168 条)
│   └── pretrain_email.jsonl    #   预训练语料 (1060 条)
└── scripts/
    ├── prepare_email_data.py   # 数据预处理 (剥离多余字段 + 校验)
    ├── train_mode1_default_sft.sh  # 手段1: Dense SFT
    ├── train_mode2_ple_sft.sh      # 手段2: PLE SFT
    ├── verify_weights.py       # 权重完整性检查
    └── eval_email.py           # 问答评估
```

## 工作流程

### 0. 数据准备

```bash
python scripts/prepare_email_data.py --src /home/EmailAgent/data/training_data
```

- **为什么需要**: EmailAgent SFT 数据带多余字段 (`task_type`/`thread_id`/`turn_count`/`time_span`),
  会被 `SFTDataset` 的严格 Features schema 拒绝 (`datasets.table.CastError: column names don't match`)。
  脚本剥离多余字段并输出纯 `{"conversations":[...]}` 格式。
- 脚本自动做 SFTDataset 加载校验 (实测通过)。
- 支持 DPO (原样) / Pretrain (原样) / 合并去重。

### 1. 训练 — 手段1 默认模式 (Dense)

```bash
bash scripts/train_mode1_default_sft.sh [hidden_size=256] [layers=6]
```

内部调用 (H1 验证参数):
```bash
cd trainer && python3 -u train_full_sft.py \
    --use_ple 0 --hidden_size 256 --num_hidden_layers 6 \
    --max_seq_len 256 --batch_size 8 --accumulation_steps 2 --epochs 3 \
    --learning_rate 2e-5 --data_path <skill>/dataset/sft_email_mixed.jsonl \
    --from_weight none --save_weight email_sft_dense_h256 --save_dir ../out
```
输出: `out/email_sft_dense_h256_256.pth` + `out/email_sft_dense_h256.log`

### 2. 训练 — 手段2 自有数据分支 (PLE)

```bash
bash scripts/train_mode2_ple_sft.sh [hidden_size=256] [layers=6] [ple_dim=96]
```

与手段1 唯一区别: `--use_ple 1 --ple_dim 96`。输出: `out/email_sft_ple_h256_256_ple.pth`。

**关键知识**:
- `use_ple` **不是** `MiniMindConfig` 构造参数, 训练器构造后设属性 `lm_config.use_ple=True` (trainer 已内置处理)
- PLE 新增参数: `ple_table` (稀疏查找表) / `ple_model_proj` / 每层 `ple_gate`/`ple_proj`/`ple_norm`
- PLE 分支从精确 no-op 开始 (`ple_norm.weight` 置零)
- 权重后缀 `_ple` (由 `trainer_utils._model_suffix` 统一计算)

### 3. 质量检查 (每次训练后必做)

```bash
# 手段1
python scripts/verify_weights.py --weight out/email_sft_dense_h256_256.pth \
    --hidden_size 256 --num_hidden_layers 6
# 手段2 (PLE)
python scripts/verify_weights.py --weight out/email_sft_ple_h256_256_ple.pth \
    --hidden_size 256 --num_hidden_layers 6 --use_ple --ple_dim 96
```

检查项:
1. 权重存在 + 大小
2. **严格加载** (missing=0 unexpected=0) — 上游 `init_model` 用 `strict=False` 静默容忍, 本检查补上 AGENTS.md 要求的严格校验
3. forward 冒烟 (logits 有限, shape 正确)

> ⚠️ PLE 权重必须带 `--use_ple` 校验, 否则报大量 missing keys (PLE 层缺失)。

### 4. 评估 (问答测试)

```bash
python scripts/eval_email.py --weight out/email_sft_dense_h256_256.pth \
    --hidden_size 256 --num_hidden_layers 6
python scripts/eval_email.py --weight out/email_sft_ple_h256_256_ple.pth \
    --hidden_size 256 --num_hidden_layers 6 --use_ple --ple_dim 96
```

- 从 `sft_email_tasks.jsonl` 按 task_type 抽样 (默认每类 2 问)
- 用 chat_template + generate 生成回答, 打印 Q/A
- 对比两种手段的回答质量 (分类/摘要/回复草稿任务)

### 5. 产物登记 (模型输出规范)

训练产出后必须登记 `docs/MODELS.md` + `CHANGELOG.md`:

```bash
python scripts/register_model.py --name "EmailAgent SFT H1 Dense" \
    --weight out/email_sft_dense_h256_256.pth \
    --data "sft_email_mixed (3432条)" --loss "<从日志读取>"
```

## 两种手段的差异与选择

| 维度 | 手段1 Dense | 手段2 PLE |
|---|---|---|
| 参数 (H1 实测) | 6.66M (d256/l6) | 10.79M (含 ple_table/proj 4.1M) |
| 权重大小 (H1) | ~16MB (fp32) | ~25MB (fp32) |
| 部署 | 通用 | ESP32-S3 (flash 驻留查找表) |
| 训练命令差异 | `--use_ple 0` | `--use_ple 1 --ple_dim 96` |
| 权重后缀 | `{save}_{dim}.pth` | `{save}_{dim}_ple.pth` |

## 完善流程 (迭代改进)

训练→评估后发现问题, 按以下顺序完善:

1. **数据问题** (回答胡言乱语/不贴合邮件): 增加数据量、检查 task_type 分布、用 `mix_medical.py` 思路混合通用数据防过拟合
2. **过拟合** (loss 低但泛化差): 降低 epochs、加 dropout、加大数据
3. **PLE 特有**: `ple_dim` 过大会增加表参数; 检查 `param_budget()` 三层预算
4. **RAFT 增强** (需证据问答): `build_medical_raft.py --no-evidence-ratio 0.3 --negative-ratio 0.15` 生成证据数据后微调
5. **部署链路** (ESP32): `quantize_ple.py` (int4 group=32) → `export_ple1.py` → `convert_h2.py` → `verify_h2.c` (PASS 阈值 maxabs<0.02)

## 验证结果记录 (2026-08-09, AMD 890M ROCm)

### 第一轮 (旧数据 3432条, lr 2e-5→5e-4 优化)

| 项 | 手段1 Dense | 手段2 PLE |
|---|---|---|
| 旧参数 (lr=2e-5, 1703条×2ep) | loss 7.30 | loss 7.26 |
| 优化后 (lr=5e-4, 400条×2ep) | loss **4.42** | loss **4.37** |
| 质量检查 | missing=0 unexpected=0 | missing=0 unexpected=0 |
| 问答输出 | 短/空 (H1 规模限制) | 短/空 (H1 规模限制) |

### 第二轮 (更新数据, 质量审计 health 84.8, lr=5e-4)

更新数据: sft_tasks 3579 / sft_threads 131 / dpo 149 / pretrain 713, PII 泄漏 0。
新增 `post_analysis.json` 质量审计报告 (a/b1/b3/e 四阶段, b1 有 P1 质量问题 7.6%)。

| 项 | 手段1 Dense | 手段2 PLE |
|---|---|---|
| 数据 (400条×2ep) | 3262 主集 | 3262 主集 |
| loss (epoch 2 后期) | **3.79** | **3.74** |
| 质量检查 | missing=0 unexpected=0 | missing=0 unexpected=0 (自动检测) |
| 问答输出 | "好的,[PERSON_165]!我们..." 连贯片段 | 同 Dense 开头一致 |

**结论**:
- 更新数据 loss 3.79/3.74 (比上轮 4.42/4.37 再降 ~15%), 问答输出从乱码/空 → 连贯中文片段
- 两手段 loss 几乎一致 + 回答一致 → 行为等价, 差异仅在 ESP32 部署场景
- 数据质量审计 (post_analysis) 可纳入管道作为前置检查 (health_score 84.8)

## 管道优化记录 (2026-08-09)

新增 `run_pipeline.py` 统一入口, 解决单步脚本分散问题:

```bash
python skills/minimind-training/scripts/run_pipeline.py --mode 1          # 手段1 全流程
python skills/minimind-training/scripts/run_pipeline.py --mode 2          # 手段2 全流程
python skills/minimind-training/scripts/run_pipeline.py --mode 1 --stage train --epochs 3
python skills/minimind-training/scripts/run_pipeline.py --mode 2 --stage verify --force
```

阶段编排: `env` (依赖/GPU/路径检查) → `data` (预处理, 幂等跳过) → `train` → `verify` → `eval`。
已解决: 依赖自动检测/安装、tee 日志目录预创建、权重存在性幂等跳过、`--force` 重跑。

## 已知陷阱

1. **SFT 数据多余字段** → CastError, 必须先 `prepare_email_data.py`
2. **`use_ple` 是事后属性**, 非 `MiniMindConfig()` 构造参数
3. **权重后缀**: dense 无后缀, PLE `_ple`, MoE `_moe`; 加载必须匹配架构
4. **脚本名 ≠ 训练器**: `wsl_train_h2_raft.sh` 实际跑 `train_full_sft.py` (SFT 微调)
5. **量化 group 必须 32** (128 崩, 16 过拟合)
6. **golden 必须来自反量化模型** (否则测的是量化误差而非转换正确性)
