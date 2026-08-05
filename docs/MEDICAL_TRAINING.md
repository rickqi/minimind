# 医疗训练说明 (Medical Training Guide)

> 本文件记录 MiniMind 医疗相关训练的完整过程、数据管线与资产。
> 覆盖: 医疗数据构建管线 (A/B1/B2/RAFT) → 训练方案 → 全部模型资产 → 部署链路。

---

## 一、医疗数据管线 (3+1 条独立管线)

### 数据源

| 源 | 内容 | 规模 |
|---|---|---|
| `esp32-ai/data_v4/corpus.txt` | 清洗后医学文本 | 348MB |
| `D:\docs\raw\medica` | 医疗指南 PDF→MD (444 文件) | 105.7MB |
| `D:\docs\raw\临床诊疗指南全集` | 41 个临床分册 | 76.5MB |
| `esp32-ai/data_v4/kb/format_data.jsonl` | 现成医学 QA | 11,000 条 |

### 管线 A: 医疗预训练语料 (`scripts/build_medical_pretrain.py`)

```
三源 → 清洗 (YAML frontmatter/页码/HTML/LaTeX/广告黑名单)
     → 段落分块 (512-1024字符, 超长硬切≤2000)
     → MinHash 去重 (字符n-gram, threshold 0.8, dedup 10.3%)
     → pretrain_medical.jsonl (128,992 条 / 333MB, ~77.8M tokens)
```

### 管线 B1: 直接转换 SFT (`scripts/build_medical_sft_b1.py`)

```
format_data.jsonl (11K QA) → 清洗 → 问题模板规范化 → sft_medical_b1.jsonl (10,681 条)
规范化: "根据临床指南,【X】内容要点" → "X的临床表现是什么？" (模板 3,449→0)
```

### 管线 B2: DeepSeek V4 Flash 合成 SFT (`scripts/build_medical_sft_b2.py`)

```
临床指南【】锚点切分 → 594 个疾病 → V4 Flash (1M上下文, JSON Output)
→ sft_medical_b2.jsonl (3,521 条, 含具体医学数值, 0 失败, 断点续跑)
```

### 管线 RAFT: 证据复述数据 (`scripts/build_medical_raft.py`)

```
format_data.jsonl 自接地 → 证据=同条目答案前60/120字符
→ sft_medical_raft.jsonl (8,000 条, 证据+问题→答案)
```

### 混合策略 (`scripts/mix_medical.py`)

```
pretrain_mixed.jsonl   = 医疗 1 : 通用 2  = 386,976 条
sft_medical_mixed.jsonl = 医疗 1 : 通用 3  = 56,808 条
```

### 质量报告 (可重复执行)

| 报告 | 位置 |
|---|---|
| 预训练语料 | `out/medical_pretrain_report.json` |
| B1 SFT | `out/medical_sft_b1_report.json` |
| B2 SFT | `out/medical_sft_b2_report.json` |
| 混合 | `out/mix_report.json` |

---

## 二、训练方案与完整过程

### 2.1 H3 混合数据从零训练 (推荐方案, 新)

**目标**: 用 minimind 标准方法, 医疗混合数据从零训练一个 38M 模型, 注入内在医学知识。

```
训练 1: 预训练 (pretrain_mixed.jsonl, 386,976 条)
  python train_pretrain.py --use_ple 1 --ple_dim 128 --hidden_size 512 \
      --num_hidden_layers 8 --max_seq_len 128 --batch_size 16 \
      --data_path ../dataset/pretrain_mixed.jsonl --save_weight pretrain_h3_mixed
  → 24,186 steps, loss 6.81 → 2.57, ~17 min (RTX 5080)

训练 2: SFT (sft_medical_mixed.jsonl, 56,808 条, 2 epochs)
  python train_full_sft.py --use_ple 1 --ple_dim 128 --hidden_size 512 \
      --num_hidden_layers 8 --max_seq_len 512 --batch_size 8 \
      --data_path ../dataset/sft_medical_mixed.jsonl \
      --from_weight pretrain_h3_mixed --save_weight full_sft_h3_mixed
  → 14,202 steps, loss 2.63 → 1.82, ~12 min

评估: 病毒性肝炎→"抗病毒治疗/控制感染"实质正确; 肺癌→咳嗽/咳痰/胸痛/咯血
结论: 内在医学知识注入成功, 但精准数值问答需 RAG+RAFT 补充
```

### 2.2 H1/H2 RAG+RAFT 方案 (已验证最优, ESP32 部署)

```
每模型: 预训练 → SFT → RAFT (证据复述微调)

H1 (d256/l6/p96, 10.79M):
  预训练 79,390 steps loss 2.27 | SFT 113,215 steps loss 2.04 | RAFT 1,000 steps loss 1.21
H2 (d384/l8/p128, 24.95M):
  预训练 79,390 steps loss 2.07 | SFT 113,215 steps loss 1.77 | DPO 4,292 steps loss 0.52 | RAFT 1,000 steps loss 1.13

评估: 无 RAG 均答错 (编造/循环); 有 RAG 均准确复述证据
  H1: "咳嗽、咳痰、咳血、胸痛" | H2: "收缩压≥140/90mmHg 即可诊断"
结论: RAG+RAFT 是小模型精准医学问答的正确路径
```

### 2.3 其他实验 (记录教训)

| 实验 | 结果 | 教训 |
|---|---|---|
| H2 医疗增强 SFT (混合) | loss 1.74 | 知识提升有限, 25M 容量瓶颈 |
| H2 纯医学 SFT (3 epochs) | loss 3.0 反弹 | 纯医学过拟合, 需混合 |
| H3 医疗增强 SFT | loss 1.56 | 38M 仍无法记忆精确医学知识 |
| H3 混合从零 (新) | loss 1.82 | ✅ 可行, 内在知识注入 |

---

## 三、全部训练资产

### 训练权重 (`D:\codes\minimind\out\`)

| 模型 | 阶段 | 文件 | 大小 |
|---|---|---|---|
| **H3 混合 (新)** | 预训练 | `pretrain_h3_mixed_512_ple.pth` | 79.1MB |
| **H3 混合 (新)** | **SFT 最终** | **`full_sft_h3_mixed_512_ple.pth`** | **79.1MB** |
| H3 | 预训练/SFT/医疗增强 | `pretrain/full_sft_h3_med_512_ple.pth` | 79.1MB |
| H2 | 预训练/SFT/DPO/RAFT | `full_sft_h2_raft_384_ple.pth` 等 | 52.3MB |
| H1 | 预训练/SFT/RAFT | `full_sft_h1_raft_256_ple.pth` | 23.7MB |

### 部署产物 (`D:\codes\esp32-ai\firmware\model_v5\`)

| 模型 | model_llm.bin | verify |
|---|---|---|
| H1 (RAFT) | 6.01MB | PASS (diff 0.00000) |
| H2 (RAFT) | 14.05MB | PASS (diff 0.00001) |

### 数据集 (`D:\codes\minimind\dataset\`)

| 文件 | 规模 |
|---|---|
| `pretrain_medical.jsonl` | 128,992 条 |
| `sft_medical_b1/b2/raft.jsonl` | 10,681 / 3,521 / 8,000 条 |
| `sft_medical_pure.jsonl` | 13,069 条 |
| `pretrain_mixed.jsonl` | 386,976 条 |
| `sft_medical_mixed.jsonl` | 56,808 条 |

### 其他资产

```
out/rag_index.pkl          jieba 医学检索索引 (11K docs)
esp32_llm_zh_v5/vocab.h    MiniMind BPE 词表 (VOCAB_N=6400)
tools/send_prompt_rag.py   PC 端 RAG 串口发送器
```

---

## 四、两条路径互补 (最终架构)

```
路径 1: H3 混合模型 (内在医学知识)
  full_sft_h3_mixed → 通用问答 + 医学泛知识
  适合: 通用对话、常识医学、无检索环境

路径 2: H1/H2 RAFT 模型 (RAG 证据复述)
  full_sft_h{1,2}_raft + rag_medical.py 检索 → 精准医学问答
  适合: ESP32 部署、精确数值/标准查询

组合: H3混合 + RAG+RAFT = 内在知识 + 检索精准度 (待验证)
```

## 五、复现命令

```bash
# 1. 构建数据管线
python scripts/build_medical_pretrain.py     # 管线A
python scripts/build_medical_sft_b1.py       # 管线B1
python scripts/build_medical_sft_b2.py --api-key $DEEPSEEK_API_KEY  # 管线B2
python scripts/build_medical_raft.py         # RAFT
python scripts/mix_medical.py                # 混合

# 2. H3 混合从零训练
bash scripts/wsl_train_h3_mixed.sh           # 预训练
bash scripts/wsl_sft_h3_mixed.sh             # SFT

# 3. H1/H2 RAFT
bash scripts/wsl_train_h1_raft.sh
bash scripts/wsl_train_h2_raft.sh

# 4. 评估
bash scripts/wsl_eval_h3_mixed.sh            # H3 混合问答
bash scripts/wsl_eval_rag_compare.sh         # H1/H2 RAG vs 无RAG
```
