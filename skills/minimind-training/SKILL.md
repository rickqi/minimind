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
├── dataset/                    # 预处理后的训练数据 (prepare_email_data.py 输出, git 不跟踪)
│   ├── sft_email_tasks.jsonl   #   任务型 SFT (3579 条)
│   ├── sft_email_threads.jsonl #   多轮对话 (131 条)
│   ├── sft_email_mixed.jsonl   #   合并去重 (3262 条)
│   ├── dpo_email.jsonl         #   DPO 偏好对 (1235 条)
│   ├── pretrain_email.jsonl    #   预训练语料 (5784 条)
│   ├── dpo_email_attach.jsonl  #   ★ 附件增强 DPO (6,799 对, build_dpo_attachment_enhanced 产出)
│   └── test_email_classify_strict.jsonl  # 分类严格测试集 (284 条)
└── scripts/
    ├── prepare_email_data.py       # 数据预处理 (剥离多余字段 + 校验)
    ├── build_attachment_index.py   # ★ 附件索引 (21GB parsed.json → 轻量 pkl)
    ├── build_dpo_attachment_enhanced.py  # ★ 附件增强 DPO 构建
    ├── train_pretrain.sh           # 预训练预热 (手段1/2 共用)
    ├── train_mode1_default_sft.sh  # 手段1: Dense SFT
    ├── train_mode2_ple_sft.sh      # 手段2: PLE SFT
    ├── verify_weights.py       # 权重完整性检查 (PLE 自动检测)
    ├── eval_email.py           # 问答评估 (含分类准确率)
    └── run_pipeline.py         # 统一管道 (env→data→train→verify→eval)
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
6. **DPO 效果不明显** (loss 降但行为不变): 检查 DPO 数据质量 — 若 rejected 多为模板/超短, 是**长度奖励坍缩**, 需附件增强或硬负样本 (见下节)

## DPO 附件增强流程 (★ 重要, 2026-08-10)

**问题背景**: EmailAgent DPO 数据 85% 是"真实回复 vs 5 字符模板", 长度比中位 20× → DPO 学到长度而非内容。同时 **51K 个附件 markdown (raw/) 从未进入训练** (利用率 0%)。

**流程** (3 步):

```bash
# 1. 构建附件索引 (21GB parsed.json → 轻量 pkl, 一次性 ~14 分钟)
python scripts/build_attachment_index.py --out out/attachment_index.pkl
#    → 23,790 个含附件 conversation_id → attachment_dir 映射

# 2. 附件增强 DPO 数据 (注入附件全文到 user_msg)
python scripts/build_dpo_attachment_enhanced.py \
    --src /home/EmailAgent/data/training_data/split/dpo_train.jsonl \
    --index out/attachment_index.pkl \
    --out skills/minimind-training/dataset/dpo_email_attach.jsonl
#    → 6,799/9,925 对 (68.5%) 注入附件上下文

# 3. DPO 重训 (新超参)
cd trainer && python3 -u train_dpo.py \
    --learning_rate 1e-6 --beta 0.3 --epochs 2 \
    --data_path ../skills/minimind-training/dataset/dpo_email_attach.jsonl \
    --from_weight email_sft_dense_h256 --save_weight email_dpo_attach_dense_h256
```

**关键参数**:
| 参数 | 旧 (无效) | 新 (有效) | 理由 |
|---|---|---|---|
| lr | 4e-8 | **1e-6** | 小模型需要更强更新 |
| beta | 0.15 | **0.3** | 更强 KL 约束防遗忘 |
| epochs | 1 | **2** | 更多学习 |

**注入格式**: `[附件: 文件名]\n附件内容` 追加到 user_msg (top-2 附件 × 2000 字符, 总长 ≤4000)。

**已验证**: 分类回归无退化 (50%), 附件问答 DPO 倾向"引用附件"回复。

**已知陷阱**:
- dpo thread_id 带 `_数字` 后缀, 需 `normalize_tid` 剥离后匹配 conversation_id
- 附件长样本使 DPO 训练慢 (AMD 890M: 2000 对×2ep ≈ 40 分钟; 6,799 对 ≈ 数小时)
- 附件 md 未脱敏, 生产需扩展 PIIMapper

**待执行任务**:
- [ ] ① 全量增强集 (6,799 对) 长训练 (验证完整效果)
- [ ] ② H2 架构重跑 (更大模型, DPO 收益更明显)
- [ ] ③ on-policy 硬负样本 (SFT 采样 rejected, 替代部分模板)

## 多场景全链训练 (v3, 2026-08-10)

EmailAgent 数据再扩充 (08-10 03:49): sft 342K/val 18K/pretrain 46K/dpo 10K, health 98.8。
分层抽样 (SFT 6000 四类均衡 + DPO 3000 + 预热 5000), 两种手段 × 三场景 (预热/SFT/DPO):

| 手段 | 预热 loss | SFT loss | DPO loss | 分类精确 (30条) |
|---|---|---|---|---|
| **手段1 Dense** | 4.47 | **0.34** | 0.62 | **60%** |
| **手段2 PLE** | 4.33 | **0.36** | 0.61 | 53% |

**质量检查**: 4 权重全部 missing=0/unexpected=0 ✅

**发现**:
- DPO (3000对) 在小模型短生成上效果不明显 (SFT/DPO 输出几乎一致) — DPO 需更大数据/更长生成才显优势
- DPO 不改变分类精度 (偏好对齐 ≠ 分类能力)
- 新验证集 (34万条数据切分) 分类难度更高: Dense 60% (vs 上轮 80% 旧验证集)

## 全量训练验证 (2026-08-09 22:30)

EmailAgent 新数据全量 (sft_train 40,680 条过滤后), 两种手段各训练:

| 手段 | 配置 | 最终 loss | 分类精确率 (30条验证集) | 质量检查 |
|---|---|---|---|---|
| **手段1 Dense** | 全量×1ep from 预热 | **1.44** | **80%** | ✅ missing=0/unexpected=0 |
| **手段2 PLE** | 全量×1ep from PLE预热 | **1.14** | 53% | ✅ missing=0/unexpected=0 |

**多场景推理** (全量模型, 验证集样本):
- 分类: 两手段均精确命中 gold (合同审核)
- 总结: 语义相关, 小模型有重复 (H1 局限)
- 回复: 模板完全正确 "收到,关于「...」一事,我们将尽快跟进处理"

**关键发现**:
- 全量数据收敛显著 (小规模 loss 4.4 → 全量 1.14/1.44)
- **全量下 Dense 分类精度 > PLE** (80% vs 53%) — 与小规模 (相当) 不同; PLE 表参数在全量训练下未体现分类优势
- Dense 训练出现 1 次瞬时 loss=nan 后自愈 (bfloat16 偶发数值尖峰)

## 新数据重训验证 (2026-08-09 21:51)

EmailAgent 数据源大幅更新 (19:51-19:59 重新生成, **量增 8x**):

| 数据集 | 旧 (skill) | 新 (数据源) | 变化 |
|---|---|---|---|
| sft_tasks | 3,577 | 29,322 | 8.2x |
| sft_threads | 131 | 1,045 | 8x |
| dpo | 149 | 1,235 | 8.3x |
| pretrain | 713 | 5,784 | 8.1x |
| **sft_train (split/)** | — | **40,881** | 新增 (EmailAgent 自动切分) |
| **B2 QA 合成** | 0 | **6,333** | 🔓 解锁 (API key 已配) |
| **RAFT 数据** | 0 | **6,333** | 🔓 解锁 |
| health_score | 84.8 | **99.0** | 质量提升 |

**用 skill 重训结果** (预热链 + 分类 2000条×3ep):
- 新数据预热 (2000条×2ep): loss 4.14
- 分类训练 (from 新预热): loss 0.29
- **独立验证集** (split/sft_val 552 条分类, 无重叠): 标签命中 **49/50=98%**, 精确匹配 **37/50=74%**

**关键改进**:
- `prepare_email_data.py` 优先导入 EmailAgent `split/sft_train.jsonl` (40,881 条全任务)
- 验证集改用 EmailAgent 官方 `split/sft_val` 切分 (更严谨, 无 hold-out 偏差)
- 严格评估用**精确匹配** (预测==gold), 非仅标签命中

## 使用指南验证记录 (2026-08-09)

三个场景逐一实测验证, 发现并修复 3 个问题:

| 场景 | 验证结果 | 修复 |
|---|---|---|
| 1. run_pipeline 全流程 | env/data/train/verify/eval 五阶段全通 | **env 阶段逻辑 bug** (--stage env 从未执行); GPU 检测 amdsmi 挂起 |
| 1. 预热链 | train_pretrain.sh 可执行, from_weight SFT 产物可评估 | sft_email_mixed_400.jsonl 重建 (run_pipeline 默认依赖) |
| 2. 分类专项 | 严格测试集 60/60 = 100% 准确率 | **eval_email 加 --all** (全量评估独立测试集) |
| 3. PLE 部署 | 4 步全通, verify PASS diff 0.00000 | 无 |

**修复内容**:
- `run_pipeline.py`: `--stage env` 时 env 实际执行; 非 env stage 自动前置 env; all 模式去重
- `run_pipeline.py`: GPU 检测改为 `is_available()` 后条件调用 `get_device_name` (规避 amdsmi 挂起)
- `eval_email.py`: 新增 `--all` 参数 (全量评估, 不抽样)
- 数据集: `sft_email_mixed_400.jsonl` 重建 (400 条)

## 验证结果记录 (2026-08-09, AMD 890M ROCm)

### 预热链优化 (②, 解决"收敛不足")

新增 `train_pretrain.sh` 预热脚本, 验证 pretrain → SFT 三段链:

| 配置 | 手段1 Dense loss |
|---|---|
| 从零 SFT (400条×2ep) | 3.79 |
| **pretrain 预热 (713条×3ep) → SFT (400条×2ep)** | **2.45** (↓35%) |

**结论**: 预热链显著提升收敛 (loss 3.79→2.45)。用法:
```bash
bash scripts/train_pretrain.sh 1 pretrain_email.jsonl 3 256 6    # 手段1 预热
bash scripts/train_mode1_default_sft.sh sft_email_mixed_400.jsonl 2 256 6 email_pretrain_1  # 预热链 SFT
```

### 评估科学性 (④, 2026-08-09)

### 发现的数据问题 (评估暴露)

EmailAgent sft_tasks 的**任务分布失衡**: 1414 条抽样训练集中分类任务仅 33 条 (2.3%),
其余是摘要/回复任务 → 混合训练模型分类准确率 **0/8** (只学会"收到,关于「...」"回复模板)。

### 解决: 分类专用训练集 + 严格 hold-out

```bash
# 从全量 1464 条分类任务切分 80/20 (无重叠)
# 训练 1140 条 / 测试 284 条
python3 构建脚本 → dataset/sft_email_classify.jsonl + test_email_classify_strict.jsonl
```

### 结果 (预热链 + 分类专用训练)

| 配置 | 训练 loss | 严格测试集准确率 |
|---|---|---|
| 混合数据 (33 条分类) | 3.38 | **0/8** |
| 分类专用 (1140 条×3ep) | 0.23 | **30/30 = 100%** |

**结论**: 任务分布是分类能力的关键; 独立测试集 (无重叠) + 定量指标 (准确率) 是评估科学性的核心。
eval_email.py 已内置分类准确率计算 (核心标签前缀匹配)。

## PLE1 部署验证 (③, 2026-08-09)

PLE 手段的完整部署链路首次跑通 (H1, email_sft_ple_h256):

| 步骤 | 命令 | 结果 |
|---|---|---|
| 量化 | `python scripts/quantize_ple.py --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 --weight email_sft_ple_h256 --save_dir out --export_dir models --group 32 --device cpu --val_iters 10 --data_path skills/minimind-training/dataset/pretrain_email.jsonl` | int4 deg **+0.0068** (误差可忽略) |
| 导出 | `python scripts/export_ple1.py --hidden_size 256 --num_hidden_layers 6 --ple_dim 96 --num_attention_heads 8 --num_key_value_heads 4 --weight email_sft_ple_h256 --save_dir out --out_dir models --seq_len 64` | PLE1 **6.31MB** (89 tensors) + golden |
| 转换 | `python chinese_v5/convert_h2.py --in ..._ple1.bin --out /tmp/model_llm.bin --bits 4` | 6.01MB, 89 tensors 匹配 |
| 验证 | `gcc verify_h2.c && /tmp/verify_h2 model_llm.bin golden.txt` | **PASS, max abs diff 0.00000** |

**结论**: PLE 部署链路完整可用, 精度无损 (C 与 PyTorch 逐位一致)。这是 PLE 手段相对 Dense 的核心价值: 训练期等价, 部署期提供 int4 量化 + PLE1 扁平格式 (适配 ESP32 flash 驻留)。

注意: `quantize_ple.py` 默认 data_path 是官方 `pretrain_t2t_mini.jsonl` (未下载会报错), 需用本地数据覆盖。

## 邮件域 RAFT 适配 (①, 2026-08-09)

医学 RAFT (`build_medical_raft.py`) 依赖知识库 QA 对 + DeepSeek B2 合成, **不适用于 EmailAgent 数据** (邮件任务无独立证据源, B2 需 API key)。

新增 `build_email_raft.py`: 邮件域 RAFT — **证据 = 邮件原文** (system 注入), 训练模型基于给定邮件执行任务:

```bash
python scripts/build_email_raft.py  # → dataset/sft_email_raft.jsonl (2000 条)
```

- 有证据样本 (70%): system 注入邮件正文 + user 任务指令 + assistant 答案
- 无证据样本 (30%): 直接给任务 (保内在能力, 防 RAFT 遗忘)
- 已验证 SFTDataset 可加载

## ROCm 生成规避 (评估发现)

AMD 890M + torch rocm 上推理有两坑, eval_email.py 已内置规避:
1. **KV cache 生成卡死** → `use_cache=False`
2. **multinomial 长序列采样死循环** → `do_sample=False` (argmax)
3. **cuda 推理不可靠** → 默认 `--device cpu` (真 NVIDIA GPU 可 `--device cuda`)

### 模型质量发现 (换行循环)

预热链模型对分类任务输出 token 234 (`\n`) 循环 — 400条×2ep SFT 不足以学会分类。**生成功能正常**, 属训练不足。生产需更多 SFT 数据 (3262 全量) + 更多 epochs。

## 验证结果记录 (2026-08-09, 第一轮+第二轮)

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
7. **DPO thread_id 带 `_数字` 后缀**: 需 `normalize_tid` 剥离后匹配 parsed.json conversation_id
8. **DPO 模板负样本 = 长度奖励坍缩**: rejected 若是 5 字符模板 (长度比 >20×), DPO 学到"长=好"而非内容; 用附件增强或 on-policy 硬负样本
9. **附件 md 未脱敏**: 注入训练数据前需扩展 PIIMapper
10. **AMD 890M 长样本 DPO 慢**: 附件注入使每条样本变长, 双模型 forward 慢 (~20s/step); 大训练集用后台跑
11. **⚠️⚠️ DPO loss 恒 0.6931 = mask 截断 (常驻调优关注项)**: `generate_loss_mask` 用 `<|im_start|>assistant\n` 匹配 assistant 段; 附件注入使 user 超长, assistant 被推到 max_seq_len 外 → **mask 全 0 → DPO 梯度为 0 → loss 恒 ln2 (静默空转, loss 看似正常但不下降)**。
    - **前置检查 (每次 DPO 训练前必做)**: `DPODataset[i]['mask_chosen'].sum()>0` (抽样 5 样本), 防静默空转
    - **修复**: max_seq_len 需覆盖 assistant 段位置 (附件注入 500 字 + max_seq_len 2048 实测有效, loss 0.69→0.046)
    - **⚠️ 非最优, 需持续验证**: 附件 500 字截断损失信息; 健康集仅 910 对 (过滤 87%); loss 0.008 可能过拟合。持续扫描 max_seq_len / 注入长度消融 / 健康集扩充 / H2 验证 (详见 docs/DPO_ANALYSIS.md §6 调优关注点)
    - **⚠️ 过拟合风险 (实测)**: 修复后 3ep 训练 loss 到 0.0000 (记忆而非泛化), 附件问答反而退化 → **建议 1-2 ep 早停** (Epoch 1 末 loss 0.008 附近泛化最佳); H1 小模型 DPO 过拟合窗口窄, 收益有限
