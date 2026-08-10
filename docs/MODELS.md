# 模型清单 (Models Inventory)

> 本文件记录 MiniMind 项目全部训练模型版本、训练数据、Loss 与部署状态。
> 更新日期: 2026-08-06 | 与 `docs/MEDICAL_TRAINING.md` 配套
>
> 演进摘要:
> 1. 基础链路 (08-03~04): H1/H2/H3 预训练 → SFT → DPO, 纯通用数据
> 2. 医疗数据管线 (08-04~05): A/B1/B2/RAFT/混合 5 条管线, 5 个医疗数据集
> 3. 医疗训练探索 (08-05): 纯医学 SFT (过拟合) → H3 混合从零 (可行) → RAG+RAFT (最优)
> 4. 组合验证 (08-05): H3混合+RAFT 无显著加成 → 精准问答靠检索, 泛知识靠内在
> 5. 部署 (08-05): H1/H2 RAFT → ESP32; H3 混合/RAFT → PC (超 ESP32 flash)

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
| **RAFT v3** | `out/full_sft_h1_raft_256_ple.pth` | 24.9MB | sft_medical_raft (8K条, 旧数据) | 1,000×3 | 1.30 → **1.21** |
| **RAFT v4** | `out/full_sft_h1_raft_v4_256_ple.pth` | 24.9MB | sft_medical_raft (8K, 负样本10%+医学过滤) | 1,000×3 | → **2.90** |

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
| **RAFT v4** | `out/full_sft_h2_raft_v4_384_ple.pth` | 54.8MB | sft_medical_raft (8K, 负样本+医学过滤) | 1,000×3 | 3.20 → **2.70** |

---

## 四、H3 系列 (d512/l8/p128, 38.16M)

| 阶段 | 权重文件 | 大小 | 训练数据 | Steps | Loss |
|---|---|---|---|---|---|
| 预训练 | `out/pretrain_h3_512_ple.pth` | 82.9MB | pretrain_t2t_mini | 79,390 | 7.04 → **1.95** |
| SFT | `out/full_sft_h3_512_ple.pth` | 82.9MB | sft_t2t_mini | 113,215 | 2.42 → **1.63** |
| 医疗增强 | `out/full_sft_h3_med_512_ple.pth` | 82.9MB | sft_medical_mixed | 7,101 | 1.47 → **1.56** |
| **混合从零预训练** | `out/pretrain_h3_mixed_512_ple.pth` | 82.9MB | pretrain_mixed (387K, 医疗1:2) | 24,186 | 6.81 → **2.57** |
| **混合 SFT** | `out/full_sft_h3_mixed_512_ple.pth` | 82.9MB | sft_medical_mixed | 7,101×2 | 2.63 → **1.82** |
| **混合+RAFT** | `out/full_sft_h3_mixed_raft_512_ple.pth` | 82.9MB | sft_medical_raft (8K) | 1,000×3 | 1.06 → **1.05** |

> H3 系列部署产物: `models/full_sft_h3_mixed_raft_h512_ple1.bin` (22.69MB PLE1) →
> `model_v5/H3-raft/model_llm.bin` (22.69MB, verify PASS)。H3 超 ESP32 flash, 适用于 PC/树莓派。

---

## 五、esp32-ai 部署产物 (firmware/model_v5)

| 模型 | 文件 | 大小 | verify | 用途 |
|---|---|---|---|---|
| **H1 RAFT v4** | `model_v5/H1/model_llm.bin` | **6.31MB** | ✅ PASS (diff 0.00000) | ESP32 轻量医学问答 |
| H1 参考 | `model_v5/H1/model.bin` | 6.31MB | — | PLE1 原始格式 (v4) |
| **H2 RAFT v4** | `model_v5/H2/model_llm.bin` | **14.73MB** | ✅ PASS (diff 0.00001) | ESP32 精准医学问答 |
| H2 参考 | `model_v5/H2/model.bin` | 14.07MB | — | PLE1 原始格式 |
| H3 混合 (参考) | `model_v5/H3/model_llm.bin` | 22.69MB | ✅ PASS (diff 0.00001) | ⚠️ 超 ESP32 flash, PC 部署 |
| H3 混合+RAFT (参考) | `model_v5/H3-raft/model_llm.bin` | 22.69MB | ✅ PASS (diff 0.00001) | ⚠️ 超 ESP32 flash, PC 部署 |

> RAG 配套: SD 卡全量索引 (113,609 docs, 医学过滤) 见下方"SD 卡全量索引"小节 — 拷入 `/sdcard/rag/` 即可供 `esp32_llm_v5_idf` 离线使用。

### H3 混合量化导出产物 (minimind/models/)

| 文件 | 大小 | 说明 |
|---|---|---|
| `full_sft_h3_mixed_h512_ple1.bin` | 22.69MB | PLE1 (混合 SFT, 无 RAFT) |
| `full_sft_h3_mixed_512_int4_g32.pth` | 40.6MB | int4 量化 (deg +0.0305) |
| `full_sft_h3_mixed_raft_h512_ple1.bin` | 22.69MB | PLE1 (**含 RAFT**, loss 1.05) |
| `full_sft_h3_mixed_raft_512_int4_g32.pth` | 40.6MB | int4 量化 (RAFT 版) |
| `full_sft_h3_mixed_h512_golden.npz/.txt` | — | golden 参考 (混合 SFT) |
| `full_sft_h3_mixed_raft_h512_golden.npz/.txt` | — | golden 参考 (RAFT) |

> ⚠️ **部署限制**: H3 (38.16M) model_llm.bin 22.69MB 超出 ESP32-S3 model 分区 (14.5MB, 16MB flash 上限)。
> H3 混合量化模型适用于 **PC/树莓派/大 flash 板卡**; ESP32 仅支持 H1 (6.31MB) / H2 (14.73MB)。
> 注: 早期导出的 H3 (21.64MB) 为**混合 SFT 版 (无 RAFT)**; 本文件已补充 **H3 混合+RAFT 版** (22.69MB)。

配套部署资产:
```
esp32_llm_zh_v5/vocab.h        MiniMind BPE 词表 (VOCAB_N=6400)
tools/send_prompt_rag.py       PC 端 RAG 串口发送器
out/rag_index.pkl              jieba 医学检索索引 (11K docs, PC 评估用)
models/full_sft_h2_raft_v4_h384_ple1.bin  H2 RAFT v4 PLE1 (14.73MB)
```

### SD 卡全量索引 (esp32 离线 RAG 部署产物, 2026-08-06 v3 医学成品)

> 供 `esp32_llm_v5_idf` 固件直接消费。固件从 `/sdcard/rag/` 读取三文件, 拷入 SD 卡同名目录即可。
> v3 版: 用 `format_data.jsonl` (11K 100% 医学成品) 构建 — 精度>召回, 10 查询对比实测 90% > 113K 广谱 80%。

| 文件 | 大小 | 说明 |
|---|---|---|
| `esp32-ai/data_v4/sd_rag/index.bin` | 2.46MB | 单字倒排 term 表 + u32 doclist (3053 terms) |
| `esp32-ai/data_v4/sd_rag/docs.bin` | 1.40MB | 证据文本 + label (10,999 docs) |
| `esp32-ai/data_v4/sd_rag/meta.bin` | 14.7KB | 单字 → u8 IDF 表 |

**部署到 esp32**: 三文件复制到 SD 卡 `/sdcard/rag/` 目录 (FAT32), 固件 `rag_sd.h` 启动时自动加载。
**数据**: `format_data.jsonl` 医学成品 (10,999 docs, 100% 医学, 含肝豆状核/白疕等修复病种)。
**质量**: 10 查询 Top-1 相关率 **90%** (肝豆状核/白疕精准命中); 高血压检索偏呼吸衰竭 (单字倒排局限)。
**注意**: `esp32_llm_zh_v5` (MM_MINIMIND) 设备端 RAG 为死代码, 此索引仅供 `esp32_llm_v5_idf` (离线 RAG 固件) 使用。

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
| `sft_medical_raft.jsonl` | 8,000 | 2.2M | RAFT 复述 (v4: 负样本+医学过滤) |
| `sft_medical_mixed.jsonl` | 56,808 | 16.6M | 医疗增强 SFT (1:3) |

---

## 七、能力定位与结论

### 模型能力矩阵 (经验证)

| 模型 | 通用对话 | 泛医学知识 | 精准医学问答 | 部署 |
|---|---|---|---|---|
| **H2 RAFT v4** | ✅ | ⚠️ | ✅ **RAG 证据复述** (盲引修复) | ESP32 model_llm.bin |
| **H1 RAFT v4** | ✅ | ⚠️ | ✅ RAG (轻量) | ESP32 model_llm.bin |
| **H3 混合** | ✅ | ✅ 内在知识 | ❌ 无检索不精准 | PC |
| H3 混合+RAFT | ✅ | ✅ | ⚠️ 组合无加成 | PC (超 ESP32 flash) |
| H2 纯医学 | ❌ 遗忘 | ⚠️ | ❌ 过拟合 | 实验 |

### ESP32 烧录状态总览

| 模型 | model_llm.bin | ESP32 flash 分区 (14.5MB) | 状态 |
|---|---|---|---|
| **H1 RAFT v4** | 6.31MB | ✅ 可烧录 | 就绪 |
| **H2 RAFT v4** | 14.73MB | ✅ 可烧录 (余量小) | 就绪 |
| H3 混合 | 22.69MB | ❌ 超限 | PC 部署 |
| H3 混合+RAFT | 22.69MB | ❌ 超限 | PC 部署 |

### 已验证结论

1. **精准医学问答 = H1/H2 RAG+RAFT**(检索质量决定精准度, 与参数规模关系小)
2. **通用+泛医学 = H3 混合模型**(内在知识注入, 从零混合训练可行)
3. **纯医学 SFT 过拟合**(loss 3.01 反弹, 需混合通用数据)
4. **RAFT 复述是格式能力**(E2 干扰修复验证, 与参数/内在知识关系小)
5. **两条路径独立成立, 组合无显著加成**

### 4 模型推理验证 (2026-08-05, `wsl_eval_4models.sh`)

| 模型 | 有RAG证据时 | 无证据/无关证据时 | 关键缺陷 |
|---|---|---|---|
| **H1 RAFT** | ✅ 精准复述 (收缩压≥140/90) | ❌ 盲目复述检索结果 | RAFT 无条件复述任何注入证据 |
| **H2 RAFT** | ✅ 精准复述 | ❌ 盲目复述/数字堆砌 | 同上 |
| **H3 混合** | — (无RAG路径) | ✅ 内在知识 (肝炎最准) + 通用对话正常 | 精准数值不足 |
| **H3 混合+RAFT** | ✅ 复述 | ❌ 盲目复述 | 综合两者弱点 |

> ⚠️ **新发现 (RAFT 盲目复述缺陷)**: RAFT 模型对**任何注入证据都无条件复述**——
> 连"介绍一下你自己"也会引用检索到的无关"减肥茶"证据。根因: 训练时系统提示
> "请根据参考资料回答" + 无条件复述奖励。**推理侧必须保证检索质量**,
> 且无检索结果时应**不注入系统提示** (否则模型会引用无关内容)。

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
bash scripts/wsl_train_h2_raft.sh   # 输出 full_sft_h2_raft_v4 (负样本+医学过滤)

# 评估
bash scripts/wsl_eval_rag_compare.sh    # H1/H2 RAG vs 无RAG
bash scripts/wsl_eval_raft_v4.sh        # RAFT v4 问答 (负样本/盲引验证)

# ESP32 部署
python scripts/export_ple1.py --weight full_sft_h2_raft_v4  # PLE1 导出
python chinese_v5/convert_h2.py --in ... --out model_llm.bin  # 转换
```

---

## 九、最近模型登记 (自动)

- **H2 RAFT v4** | `out/full_sft_h2_raft_v4_384_ple.pth` | 54.85MB | sft_medical_raft (8K, 负样本+医学过滤) | - | **2.70**
-   说明: PLE1: `models/full_sft_h2_raft_v4_h384_ple1.bin` (14.73MB) | int4: `models/full_sft_h2_raft_v4_384_int4_g32.pth` (26.58MB) | 部署: `../esp32-ai/firmware/model_v5/H2/model_llm.bin` (14.73MB) | verify PASS diff 0.00001
- **H1 RAFT v4** | `out/full_sft_h1_raft_v4_256_ple.pth` | 24.88MB | sft_medical_raft (8K, 负样本+医学过滤) | - | **2.90**
  - 说明: PLE1: `models/full_sft_h1_raft_v4_h256_ple1.bin` (6.31MB) | 部署: `../esp32-ai/firmware/model_v5/H1/model_llm.bin` (6.31MB) | verify PASS diff 0.00000
- **EmailAgent pretrain H1 (预热)** | `out/email_pretrain_1_256.pth` | 16.61MB | pretrain_email v2 (2000条×2ep) | - | **4.14**
- **EmailAgent 分类 SFT H1 (v2数据)** | `out/email_sft_dense_h256_256.pth` | 16.61MB | sft classify v2 (2000条×3ep) | - | **0.29**
-   说明: verify 严格测试集 精确 74%
- **EmailAgent 全量 Dense H1 (40680条)** | `out/email_sft_dense_h256_256.pth` | 16.61MB | sft_train 全量 (40680条×1ep, from 预热) | - | **1.44**
-   说明: verify 分类精确 80% (30条验证集)
- **EmailAgent 全量 PLE H1 (40680条)** | `out/email_sft_ple_h256_256_ple.pth` | 24.88MB | sft_train 全量 (40680条×1ep, from PLE预热) | - | **1.14**
-   说明: verify 分类精确 53% (30条验证集)
- **EmailAgent v3 Dense SFT H1 (6000条多场景)** | `out/email_sft_dense_h256_256.pth` | 16.61MB | sft v3 分层抽样 (6000条×3ep) | - | **0.34**
-   说明: verify 分类精确 60%
- **EmailAgent v3 Dense DPO H1** | `out/email_dpo_dense_h256_256.pth` | 16.61MB | dpo v3 (3000对×2ep) | - | **0.62**
-   说明: verify 分类精确 60%
- **EmailAgent v3 PLE SFT H1 (6000条多场景)** | `out/email_sft_ple_h256_256_ple.pth` | 24.88MB | sft v3 分层抽样 (6000条×3ep) | - | **0.36**
-   说明: verify 分类精确 53%
- **EmailAgent v3 PLE DPO H1** | `out/email_dpo_ple_h256_256_ple.pth` | 24.88MB | dpo v3 (3000对×2ep) | - | **0.61**
-   说明: verify 分类精确 53%
- **EmailAgent 附件增强 DPO H1 (2000对)** | `out/email_dpo_attach_dense_h256_256.pth` | 16.61MB | dpo 附件增强 (2000对×2ep, lr 1e-6/beta 0.3) | - | **0.55-0.69**
-   说明: verify 分类回归 50% (无退化); 附件问答引用附件
- **EmailAgent 修复版 DPO H1 (mask截断修复)** | `out/email_dpo_healthy_h256_256.pth` | 16.61MB | 健康集 910对×3ep, max_seq 2048, lr 5e-6/beta 0.5 | - | **0.69→0.0000 (过拟合)**
-   说明: verify 分类回归 50% (无退化)
- **EmailAgent H2 SFT (d384, 全量3262条)** | `out/email_sft_dense_h384_384.pth` | 39.37MB | sft_email_mixed (3262条×3ep) | - | **1.13**
-   说明: verify 分类 45%; 附件问答自然 (vs H1 重复)
