# Changelog

本文件记录 MiniMind 仓库的显著变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### 🎉 新增

- **PLE (Per-Layer Embedding) 架构支持**（`[feat] PLE per-layer embedding support`）
  - `model/model_minimind.py` 新增 `use_ple` / `ple_dim` 配置与完整 PLE 模块，参考 esp32-ai（Gemma-4 PLE）实现：
    - 每层残差注入 `Embedding(vocab, n_layers×ple_dim)` 稀疏查找表（`ple_table`），以 flash 存储换取 SRAM 驻留空间
    - 上下文感知投影 `ple_model_proj` + 逐层 gate/proj/norm 支路
    - `post_init()` 后置零 `ple_norm` 增益，使 PLE 分支从精确 no-op 开始
  - `MiniMindForCausalLM.param_budget()`：core/table/stream 三层参数预算（esp32-ai 设计思想），用于嵌入式部署容量评估
  - `trainer/trainer_utils.py` 新增 `_model_suffix()`，PLE/MoE/Dense 权重后缀独立（`_ple`/`_moe`/``），避免互相覆盖
  - `trainer/train_pretrain.py`、`trainer/train_full_sft.py` 新增 `--use_ple` / `--ple_dim` 参数
  - `eval_llm.py` 新增 `--use_ple` / `--ple_dim` 参数，标准推理入口可加载 PLE 权重
- **int4 量化导出脚本** `scripts/quantize_ple.py`（移植 esp32-ai `src/quantize.py` 的 group-wise symmetric int4 PTQ）
  - `--group 32` 适配 SFT 模型（esp32-ai 实测 group=128 会崩、group=32 可用）
  - 输出量化前后 val loss 退化（deg）报告 + int4 codes/scales 权重
  - `--export_dir`（默认 `models/`）：部署产物与训练产物 `out/` 分离存放（`models/` 已 gitignore）

### 🚀 训练成果（实验性，未随仓库发布权重）

| 模型 | 配置 | 总参数 | int4 尺寸 | PLE1 尺寸 | 预训练 loss | SFT loss | int4 deg |
|---|---|---|---|---|---|---|---|
| H1 | d256/l6/p96 | 10.79M | 5.4MB | 6.09MB | 2.27 | 2.04 | +0.124 |
| H2 | d384/l8/p128 | 24.95M | 12.5MB | 14.07MB | 2.07 | 1.77 | +0.041 |
| H3 | d512/l8/p128 | 38.16M | 19.1MB | 21.51MB | 1.95 | 1.63 | +0.033 |

- 在 WSL（RTX 5080）上使用 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl` 训练，总耗时约 1 小时（H1）/ 2 小时（H2）/ 2.5 小时（H3）
- 权重文件：`out/pretrain_h{1,2,3}_{dim}_ple.pth`、`out/full_sft_h{1,2,3}_{dim}_ple.pth`（fp16，gitignored）
- int4 导出：`models/full_sft_h{1,2,3}_{dim}_int4_g32.pth`（gitignored，codes+fp16 scales）
  - H1 实测 11.5 MB（理论纯 int4 权重 5.4 MB）
  - H2 实测 26.5 MB（理论纯 int4 权重 12.5 MB）
  - H3 实测 40.6 MB（理论纯 int4 权重 19.1 MB）
- **PLE1 扁平二进制导出** `scripts/export_ple1.py`（对齐 esp32-ai `src/export.py` 格式，供 C 运行时 mmap 烧录）
  - Header: magic `0x504C4531` + 8×int32（vocab/d/layers/heads/ffn/ple_dim/seq_len/group）+ float rope_theta
  - Tensor 布局：int4 codes 2-per-byte（ragged, group=32）+ fp16 scales；norms 保持 fp32
  - 导出 golden 参考（固定 prompt 的最后位置 logits，npz+txt）供 C 端口正确性验证
  - 产物：`models/full_sft_h{1,2,3}_{dim}_ple1.bin`（H1 **6.09 MB** / H2 **14.07 MB** / H3 **21.51 MB**）+ `_golden.npz/_golden.txt`
- **DPO 偏好优化**（H1/H2/H3, `trainer/train_dpo.py` 新增 `--use_ple` 支持）
  - 分别在 `full_sft_h{1,2,3}` 上叠加 DPO（beta=0.15, lr=4e-8, dpo.jsonl 17,166 条偏好对, 4,292 steps）
  - DPO loss：H1 0.68→0.51 / H2 0.68→0.52 / H3 0.68→0.57
  - 效果：消除重复性表达、回答更简洁准确、结构化更强
  - 权重：`out/dpo_h{1,2,3}_{dim}_ple.pth`（fp16，gitignored）
- **DPO 后量化导出**（针对 PLE 模式的最终部署产物, 基于 DPO 调优权重）
  - int4：`models/dpo_h{1,2,3}_{dim}_int4_g32.pth`（H1 11.5MB / H2 26.5MB / H3 40.6MB）
  - PLE1：`models/dpo_h{1,2,3}_{dim}_ple1.bin`（H1 **6.09 MB** / H2 **14.07 MB** / H3 **21.51 MB**）+ golden
  - DPO 后量化鲁棒性与 SFT 后一致（deg H1 +0.124 / H2 +0.041 / H3 +0.033）——极小 lr 偏好微调不改变基础能力
- **ESP32 词表生成** `scripts/gen_vocab_minimind.py`（MiniMind BPE+ByteLevel → esp32-ai vocab.h）
  - 用 `bytes_to_unicode` **逆映射**还原每个 token 的原始 UTF-8 字节（关键：不用 `tok.decode([i])`，避免 ByteLevel 中间片段被 U+FFFD 替换符污染）
  - 产物：`esp32-ai/firmware/esp32_llm_zh_v3/vocab.h`（**VOCAB_N=6400**, blob 29.5KB）
  - 验证：特殊 token 与 BPE token（"你好"/"什么"/"是"）字节往返正确
- **`export_ple1.py` GQA→MHA 转换**（`--num_key_value_heads`）
  - kv 头 `repeat_interleave` 复制扩展（4→8 头），数学等价（实测 logits diff=0.0）
  - 使 PLE1 导出兼容 MHA 风格的 C 推理核（esp32-ai llm.h）
- **医疗数据补充管线**（三管线独立执行 + 混合策略）
  - **管线A** `scripts/build_medical_pretrain.py`: 医疗 Pretrain 语料
    - 三源: `esp32-ai/data_v4/corpus.txt` (348MB) + `D:\docs\raw\medica` (444 md) + `D:\docs\raw\临床诊疗指南全集` (76 md)
    - 清洗（YAML frontmatter/页码/HTML/LaTeX/水印黑名单）+ 段落分块 + **字符 n-gram MinHash 去重**（threshold 0.8, dedup 9.5%）
    - 产物: `dataset/pretrain_medical.jsonl`（**123,292 条** / 348.9MB）+ `out/medical_pretrain_report.json`
  - **管线B1** `scripts/build_medical_sft_b1.py`: 直接转换
    - `esp32-ai/data_v4/kb/format_data.jsonl`（11K 可读医学 QA）→ minimind SFT 格式
    - 产物: `dataset/sft_medical_b1.jsonl`（**10,683 条**）
  - **管线B2** `scripts/build_medical_sft_b2.py`: DeepSeek V4 Flash 合成 QA
    - 临床诊疗指南【】锚点切分 → 提取 **594 个疾病**（概述/临床表现/诊断要点/治疗原则及方案）
    - V4 Flash（1M 上下文, JSON Output）批量生成, 断点续跑缓存, API timeout 保护
    - 产物: `dataset/sft_medical_b2.jsonl`（**3,521 条**, 0 失败, 答案含具体医学数值）
  - **混合策略** `scripts/mix_medical.py`（esp32-ai V4 经验调整）
    - Pretrain 1:2（医学:通用）→ `dataset/pretrain_mixed.jsonl`（**369,876 条** / 589.4MB）
    - SFT 1:3 → `dataset/sft_medical_mixed.jsonl`（**56,816 条** / 92.4MB）
  - 三管线均输出质量报告（可重复执行对比）, B2 支持断点续跑
- **医疗增强训练验证**（H2 PLE, WSL RTX 5080）
  - 从 `full_sft_h2` 用 `sft_medical_mixed.jsonl`（56,816 条）微调, lr=2e-5, 1 epoch
  - **7,102 steps, loss 1.7353**（原 full_sft_h2 为 2.04, 医疗数据被吸收）
  - 权重: `out/full_sft_h2_med_384_ple.pth`（54.8MB, fp16）
  - **评估结论**: 医学知识提升有限（H2 24.95M 参数容量不足 + 混合数据 75% 通用样本稀释 + B1 模板化问题）。与 esp32-ai 结论一致: 小模型医学能力瓶颈在数据多样性与参数容量, 非训练量
- **数据质量改进**（基于分析报告的执行）
  - **改进1 清洗强化**（`build_medical_pretrain.py`）:
    - 广告短语黑名单扩充（"帮助了上万人"/"带书签索引"/"电子书代找"/版权页等）
    - 超长块兜底硬切: `max_len` 从 **74,226 → 2000** 字符
    - 重跑后: 广告噪声 **0 命中**, 超长块 **0%**, 总噪声率 2.3%（仅 www/版权页低影响残留）
    - 产物更新: `pretrain_medical.jsonl` **128,992 条**（dedup 10.3%, avg 979 字符）
  - **改进2 B1 问题规范化**（`build_medical_sft_b1.py`）:
    - "根据临床指南,【X】的内容要点有哪些" 模板 → 自然问句（"X的临床表现是什么？"等）
    - 模板残留 **3,449 → 0**, 非问句占比 45.8% → 38.9%（剩余为"XX的临床诊疗要点是什么"自然形式）
    - 产物更新: `sft_medical_b1.jsonl` **10,681 条**
  - **改进2b B1 尾缀优化**:
    - "XX的临床诊疗要点是什么" 尾缀 → "XX的诊疗要点有哪些？"（3,449 → **0**）
    - 自然问句形式达 **94%**
  - **纯医学 SFT 数据集** `scripts/build_medical_sft_pure.py`:
    - B1 过滤残缺/模板/超长（1,133 条剔除） + B2 全量 = **13,069 条**纯医学 QA（无通用稀释）
  - **混合产物重新生成**（基于改进后数据）:
    - `pretrain_mixed.jsonl`: 医学 128,992 + 通用 257,984 = **386,976 条**
    - `sft_medical_mixed.jsonl`: 医学 14,202 + 通用 42,606 = **56,808 条**
- **纯医学 SFT 训练验证**（H2, 3 epochs, 13,069 条纯医学）:
  - 权重: `out/full_sft_h2_pure_384_ple.pth`
  - **结论**: 医学问答未提升且出现灾难性遗忘（过拟合医学格式 + 丢失通用能力）
  - **最终结论确认**: H2 24.95M 参数容量是医学知识**硬瓶颈**——纯 SFT 无法注入医学知识, 小模型需 RAG 或更大模型（与 esp32-ai 结论一致）
- **路径验证: H3 训练 vs RAG 方案**
  - **路径1 H3 医疗增强**（`wsl_train_h3_med.sh`, full_sft_h3 + sft_medical_mixed）:
    - loss 1.56, 但医学问答仍无法达标（38.16M 容量仍不足, 仅高血压/肺癌部分沾边）
  - **路径2 H1/H2 RAG**（`rag_medical.py` + `build_medical_raft.py` + `wsl_train_h2_raft.sh`）:
    - **KB 索引**: format_data.jsonl (11K 医学 QA) → **jieba 分词倒排索引** (29,849 terms) + IDF 加权
    - **检索**: IDF 加权 Top-2 证据注入 ChatML（esp32-ai: Top-1 会退化）
    - **RAFT 微调**: 8,000 条"证据+问题→答案"自接地数据, H2 微调 3 epochs (loss 1.13)
    - **验证成功**: 高血压→"收缩压≥140/90mmHg 即可诊断"; 肺癌早期→准确复述症状; 感染性休克→体循环阻力/酸中毒
    - **结论**: RAG+RAFT 是让小模型用上医学知识的**正确路径**（对比裸模型循环重复/编造）
  - **最终对比**: 裸模型 < SFT < H3 SFT < **H2+RAG+RAFT**（唯一能给出准确医学标准者）
- **H1/H2 RAG vs 无 RAG 对比评估**（`wsl_eval_rag_compare.sh`, 5 医学问题 × 2 模型 × 2 模式）:
  - **无 RAG**: H1/H2 全部答错（编造/循环/泛泛）——小模型无外部知识必然失败
  - **有 RAG**: H1/H2 **均准确复述证据**——H1 (10.79M) 也能给出"咳嗽、咳痰、咳血、胸痛"和"收缩压≥140/90mmHg"
  - **结论**: RAG 效果决定性, 双模型 RAFT 复述能力都达标（H1 不逊 H2）, 剩余差异仅在 KB 检索精准度
- **ESP32 V5 固件 RAG 链路**（esp32-ai `esp32_llm_zh_v5.ino` + `tools/send_prompt_rag.py`）:
  - 固件修复: ChatML 特殊 token（屏蔽 im_start=1, 停止 endoftext=0/im_end=2）; `MM_MINIMIND` 禁用设备端 char-level RAG
  - PC 端发送器: jieba IDF 检索 (11K QA) → ChatML 证据注入 (≤100 tokens) → 串口 `{"ids":[...]}` → 生成回传


### 待办

- [ ] 更大模型（H3 38.16M）医疗增强或纯医学 SFT（不分混合）重试
