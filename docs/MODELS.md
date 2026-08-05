# 模型清单 (Models Inventory)

> 本文件记录 MiniMind 项目全部训练模型版本、训练数据、Loss 与部署状态。
> 更新日期: 2026-08-05 | 与 `docs/MEDICAL_TRAINING.md` 配套

---

## 一、架构总览

三个 PLE (Per-Layer Embedding) 架构, 参数随规模递增:

| 架构 | 配置 | 总参数 | core | table | stream |
|---|---|---|---|---|---|
| H1 | d256/l6/p96 | **10.79M** | 5.46M | 3.69M | 1.64M |
| H2 | d384/l8/p128 | **24.95M** | 15.94M | 6.55M | 2.46M |
| H3 | d512/l8/p128 | **38.16M** | 28.33M | 6.55M | 3.28M |

- 词表: 6400 (MiniMind BPE)
- 训练环境: WSL Ubuntu-22.04 + RTX 5080 (16GB), torch 2.9.1+cu128
- 权重格式: fp16 PyTorch `.pth`

---

## 二、H1 系列 (d256/l6/p96, 10.79M)

| 阶段 | 权重文件 | 大小 | 训练数据 | Steps | Loss |
|---|---|---|---|---|---|
| 预训练 | `out/pretrain_h1_256_ple.pth` | 24.9MB | pretrain_t2t_mini (1.27M条) | 79,390 | 7.10 → **2.27** |
| SFT | `out/full_sft_h1_256_ple.pth` | 24.9MB | sft_t2t_mini (905K条) | 113,215 | 2.76 → **2.04** |
| DPO | `out/dpo_h1_256_ple.pth` | 24.9MB | dpo.jsonl (17K对) | 4,292 | 0.66 → **0.51** |
| **RAFT** | `out/full_sft_h1_raft_256_ple.pth` | 24.9MB | sft_medical_raft (8K条) | 1,000×3 | 1.30 → **1.21** |

---

## 三、H2 系列 (d384/l8/p128, 24.95M)

| 阶段 | 权重文件 | 大小 | 训练数据 | Steps | Loss |
|---|---|---|---|---|---|
| 预训练 | `out/pretrain_h2_384_ple.pth` | 54.8MB | pretrain_t2t_mini | 79,390 | 7.01 → **2.07** |
| SFT | `out/full_sft_h2_384_ple.pth` | 54.8MB | sft_t2t_mini | 113,215 | 2.53 → **1.77** |
| DPO | `out/dpo_h2_384_ple.pth` | 54.8MB | dpo.jsonl | 4,292 | 0.66 → **0.52** |
| 医疗增强 | `out/full_sft_h2_med_384_ple.pth` | 54.8MB | sft_medical_mixed (57K) | 7,102 | 1.82 → **1.74** |
| 纯医学 | `out/full_sft_h2_pure_384_ple.pth` | 54.8MB | sft_medical_pure (13K) | 1,634×3 | 3.52 → **3.01** (过拟合↑) |
| **RAFT v3** | `out/full_sft_h2_raft_v3_384_ple.pth` | 54.8MB | sft_medical_raft (8K, E2干扰) | 1,000×3 | 2.57 → **2.25** |

---

## 四、H3 系列 (d512/l8/p128, 38.16M)

| 阶段 | 权重文件 | 大小 | 训练数据 | Steps | Loss |
|---|---|---|---|---|---|
| 预训练 | `out/pretrain_h3_512_ple.pth` | 82.9MB | pretrain_t2t_mini | 79,390 | 7.04 → **1.95** |
| SFT | `out/full_sft_h3_512_ple.pth` | 82.9MB | sft_t2t_mini | 113,215 | 2.42 → **1.63** |
| 医疗增强 | `out/full_sft_h3_med_512_ple.pth` | 82.9MB | sft_medical_mixed | 7,101 | 1.47 → **1.56** |
| **混合从零预训练** | `out/pretrain_h3_mixed_512_ple.pth` | 82.9MB | pretrain_mixed (387K, 医疗1:2) | 24,186 | 6.81 → **2.57** |
| **混合 SFT** | `out/full_sft_h3_mixed_512_ple.pth` | 82.9MB | sft_medical_mixed | 7,101×2 | 2.63 → **1.82** |
| 混合+RAFT | `out/full_sft_h3_mixed_raft_512_ple.pth` | 82.9MB | sft_medical_raft | 1,000×3 | 1.06 → **1.05** |

---

## 五、esp32-ai 部署产物 (firmware/model_v5)

| 模型 | 文件 | 大小 | verify | 用途 |
|---|---|---|---|---|
| **H1 RAFT** | `model_v5/H1/model_llm.bin` | **6.31MB** | ✅ PASS (diff 0.00000) | ESP32 轻量医学问答 |
| H1 参考 | `model_v5/H1/model.bin` | 6.09MB | — | PLE1 原始格式 |
| **H2 RAFT** | `model_v5/H2/model_llm.bin` | **14.73MB** | ✅ PASS (diff 0.00001) | ESP32 精准医学问答 |
| H2 参考 | `model_v5/H2/model.bin` | 14.07MB | — | PLE1 原始格式 |
| H3 混合 (参考) | `model_v5/H3/model_llm.bin` | 21.64MB | ✅ PASS (diff 0.00001) | ⚠️ 超 ESP32 flash, PC 部署 |
| H3 混合+RAFT (参考) | `model_v5/H3-raft/model_llm.bin` | 21.64MB | ✅ PASS (diff 0.00001) | ⚠️ 超 ESP32 flash, PC 部署 |

### H3 混合量化导出产物 (minimind/models/)

| 文件 | 大小 | 说明 |
|---|---|---|
| `full_sft_h3_mixed_h512_ple1.bin` | 22.69MB | PLE1 (混合 SFT, 无 RAFT) |
| `full_sft_h3_mixed_512_int4_g32.pth` | 40.6MB | int4 量化 (deg +0.0305) |
| `full_sft_h3_mixed_raft_h512_ple1.bin` | 22.69MB | PLE1 (**含 RAFT**, loss 1.05) |
| `full_sft_h3_mixed_raft_512_int4_g32.pth` | 40.6MB | int4 量化 (RAFT 版) |
| `full_sft_h3_mixed_h512_golden.npz/.txt` | — | golden 参考 (混合 SFT) |
| `full_sft_h3_mixed_raft_h512_golden.npz/.txt` | — | golden 参考 (RAFT) |

> ⚠️ **部署限制**: H3 (38.16M) model_llm.bin 21.64MB 超出 ESP32-S3 model 分区 (14.5MB, 16MB flash 上限)。
> H3 混合量化模型适用于 **PC/树莓派/大 flash 板卡**; ESP32 仅支持 H1 (6.31MB) / H2 (14.73MB)。
> 注: 早期导出的 H3 (21.64MB) 为**混合 SFT 版 (无 RAFT)**; 本文件已补充 **H3 混合+RAFT 版**。

配套部署资产:
```
esp32_llm_zh_v5/vocab.h        MiniMind BPE 词表 (VOCAB_N=6400)
tools/send_prompt_rag.py       PC 端 RAG 串口发送器
out/rag_index.pkl              jieba 医学检索索引 (11K docs)
```

---

## 六、训练数据资产 (dataset/)

| 数据集 | 条数 | tokens | 用途 |
|---|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1,270,238 | ~162M | H1/H2/H3 基础预训练 |
| `sft_t2t_mini.jsonl` | 905,718 | ~17M | 基础 SFT |
| `pretrain_medical.jsonl` | 128,992 | 77.8M | 医疗预训练 (管线A) |
| `pretrain_mixed.jsonl` | 386,976 | 140M | H3 混合从零 (医疗1:2) |
| `sft_medical_b1.jsonl` | 10,681 | 1.8M | 医学 SFT (直接转换) |
| `sft_medical_b2.jsonl` | 3,521 | 0.3M | 医学 SFT (V4 Flash 合成) |
| `sft_medical_pure.jsonl` | 13,069 | 1.9M | 纯医学 SFT |
| `sft_medical_raft.jsonl` | 8,000 | 2.2M | RAFT 复述 (E2 干扰) |
| `sft_medical_mixed.jsonl` | 56,808 | 16.6M | 医疗增强 SFT (1:3) |

---

## 七、能力定位与结论

### 模型能力矩阵 (经验证)

| 模型 | 通用对话 | 泛医学知识 | 精准医学问答 | 部署 |
|---|---|---|---|---|
| **H2 RAFT v3** | ✅ | ⚠️ | ✅ **RAG 证据复述** | ESP32 model_llm.bin |
| **H1 RAFT** | ✅ | ⚠️ | ✅ RAG (轻量) | ESP32 model_llm.bin |
| **H3 混合** | ✅ | ✅ 内在知识 | ❌ 无检索不精准 | PC |
| H3 混合+RAFT | ✅ | ✅ | ⚠️ 组合无加成 | 实验 |
| H2 纯医学 | ❌ 遗忘 | ⚠️ | ❌ 过拟合 | 实验 |

### 已验证结论

1. **精准医学问答 = H1/H2 RAG+RAFT**(检索质量决定精准度, 与参数规模关系小)
2. **通用+泛医学 = H3 混合模型**(内在知识注入, 从零混合训练可行)
3. **纯医学 SFT 过拟合**(loss 3.01 反弹, 需混合通用数据)
4. **RAFT 复述是格式能力**(E2 干扰修复验证, 与参数/内在知识关系小)
5. **两条路径独立成立, 组合无显著加成**

### 对比参照 (esp32-ai 从零训练)

| 模型 | 参数 | tokens | val PPL |
|---|---|---|---|
| esp32-ai zh5 | 13.7M | 99.3M 医疗 | 12.13 |
| esp32-ai zh7 | 16.4M | 130M 医疗 | 11.59 |
| MiniMind H3 混合 | 38.16M | 140M 混合 | loss 2.57 (≈ppl 13) |

---

## 八、复现命令速查

```bash
# H1/H2/H3 基础训练
bash scripts/wsl_train_h1.sh        # 预训练
bash scripts/wsl_sft_h1.sh          # SFT

# 医疗数据管线
python scripts/build_medical_pretrain.py
python scripts/build_medical_sft_b1.py
python scripts/build_medical_sft_b2.py --api-key $DEEPSEEK_API_KEY
python scripts/build_medical_raft.py --no-evidence-ratio 0.3
python scripts/mix_medical.py

# H3 混合从零
bash scripts/wsl_train_h3_mixed.sh
bash scripts/wsl_sft_h3_mixed.sh

# RAFT 微调
bash scripts/wsl_train_h1_raft.sh
bash scripts/wsl_train_h2_raft.sh   # 输出 full_sft_h2_raft_v3

# 评估
bash scripts/wsl_eval_rag_compare.sh    # H1/H2 RAG vs 无RAG
bash scripts/wsl_eval_raft_v3.sh        # RAFT v3 问答

# ESP32 部署
python scripts/export_ple1.py --weight full_sft_h2_raft_v3  # PLE1 导出
python chinese_v5/convert_h2.py --in ... --out model_llm.bin  # 转换
```
