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
- **DPO 偏好优化**（H2, `trainer/train_dpo.py` 新增 `--use_ple` 支持）
  - 在 `full_sft_h2` 上叠加 DPO（beta=0.15, lr=4e-8, dpo.jsonl 17,166 条偏好对, 4,292 steps）
  - DPO loss 0.68 → 0.52
  - 效果：消除重复性表达、回答更简洁准确、结构化更强
  - 权重：`out/dpo_h2_384_ple.pth`（fp16，gitignored）

### 待办

- 无（PLE 全流程：预训练 → SFT → DPO → int4 量化 → PLE1 导出 已完成）
