# AGENTS.md

> 本文件为 OpenCode/Claude 等 AI 代理提供本仓库的关键上下文, 避免踩坑。
> 语言: 中文 (与仓库内 README.md / docs/*.md 一致)

---

## 仓库定位

**MiniMind** — 从 0 训练极小型 LLM 的完整链路项目 (PLE 架构, Per-Layer Embedding)。
包含: 模型训练 (PyTorch 原生, WSL-GPU)、医疗数据管线、量化导出、ESP32 部署产物生成与主机验证。

- 训练环境: **WSL Ubuntu-22.04 + RTX 5080** (torch 2.9.1+cu128), 训练脚本在 `scripts/wsl_*.sh`
- 模型架构: H1 (d256/l6/p96) / H2 (d384/l8/p128) / H3 (d512/l8/p128), 词表 6400 (BPE)
- 医疗数据: 5 条管线 (A/B1/B2/RAFT/混合), 详见 `docs/MEDICAL_TRAINING.md`
- 模型清单: 全部训练/部署产物登记在 `docs/MODELS.md`

---

## 🔥 模型输出规范 (硬性要求, 每次模型文件输出必须执行)

**任何操作生成/更新了模型权重、量化产物、部署二进制或 golden 文件后, 必须同步完成以下两项, 否则视为未完成:**

### 1. 更新 `CHANGELOG.md` (变更说明)

- 在 Unreleased 区追加条目, 记录:
  - 模型/版本名 + 权重路径
  - 训练数据 + Steps + Loss (从日志读取)
  - 量化 deg / verify 结果 (如有)
  - 部署产物路径
- 格式示例:
  ```markdown
  - **H2 RAFT v4** (`out/full_sft_h2_raft_v4_384_ple.pth`, 54.8MB):
    - 训练: sft_medical_raft (8K, 负样本10%+医学过滤), 3 epochs, loss 2.70
    - 导出: `models/full_sft_h2_raft_v4_h384_ple1.bin` (14.73MB) + int4 (26.6MB)
    - 部署: `esp32-ai/firmware/model_v5/H2/model_llm.bin` (verify PASS diff 0.00001)
  ```

### 2. 更新 `docs/MODELS.md` (模型清单 + 输出物位置)

- 在对应架构系列表追加/更新该版本行 (权重文件/大小/数据/Steps/Loss)
- 在部署产物区更新 model_llm.bin / golden 位置
- 新增量化导出产物时, 在对应小节登记: 文件路径 + 大小 + 说明
- **必须包含完整输出物位置** (相对路径), 便于追踪

### 3. 登记工具 (推荐使用)

```bash
# 自动登记模型产物到 docs/MODELS.md + CHANGELOG.md
python scripts/register_model.py --name "H2 RAFT v4" \
    --weight out/full_sft_h2_raft_v4_384_ple.pth \
    --ple1 models/full_sft_h2_raft_v4_h384_ple1.bin \
    --int4 models/full_sft_h2_raft_v4_384_int4_g32.pth \
    --deploy ../esp32-ai/firmware/model_v5/H2/model_llm.bin \
    --data "sft_medical_raft (8K, 负样本)" --loss "2.70"
```

---

## 多环境隔离 (重要)

| 环境 | 训练脚本 | 权重 | 部署产物 | 词表 |
|---|---|---|---|---|
| 本地训练 | `trainer/*.py` | `out/*.pth` | `models/*.bin` | `model/tokenizer.json` |
| ESP32 部署 | `scripts/export_ple1.py` + esp32-ai `convert_h2.py` | `models/*_ple1.bin` | `esp32-ai/firmware/model_v5/{H1,H2,H3}/` | `esp32_llm_zh_v5/vocab.h` |

**规则**:
- 训练权重只存 `out/` (gitignored), 部署产物登记到 `models/` (gitignored)
- ESP32 产物通过 convert_h2.py 生成到 esp32-ai 仓库, 需在 esp32-ai 侧提交
- **本项目只做模型生成 + 主机验证, 不烧录/实机验证** (COM 口操作不属于本项目职责)

---

## 精确命令速查

### 医疗数据管线
```bash
python scripts/build_medical_pretrain.py                    # 管线A: 预训练语料
python scripts/build_medical_sft_b1.py                      # 管线B1: 直接转换
python scripts/build_medical_sft_b2.py --api-key $KEY       # 管线B2: V4 Flash 合成
python scripts/build_medical_raft.py --no-evidence-ratio 0.3 --negative-ratio 0.15 --med-only
python scripts/mix_medical.py                               # 混合 (1:2 / 1:3)
```

### 训练 (WSL GPU)
```bash
bash scripts/wsl_train_h2_raft.sh        # H2 RAFT 微调 (输出 full_sft_h2_raft_v4)
bash scripts/wsl_train_h3_mixed.sh       # H3 混合从零预训练
bash scripts/wsl_sft_h3_mixed.sh         # H3 混合 SFT
```

### 评估
```bash
bash scripts/wsl_eval_raft_v4.sh         # RAFT v4 问答 (负样本/盲引验证)
bash scripts/wsl_eval_rag_compare.sh     # H1/H2 RAG vs 无RAG
bash scripts/wsl_eval_4models.sh         # 4 模型横向对比
```

### ESP32 部署产物生成 (只生成+验证, 不烧录)
```bash
python scripts/export_ple1.py --weight full_sft_h2_raft_v4 --num_attention_heads 8 --num_key_value_heads 4
# -> models/full_sft_h2_raft_v4_h384_ple1.bin + golden
# 转 ESP32 格式 (esp32-ai 仓库)
python chinese_v5/convert_h2.py --in ... --out firmware/model_v5/H2/model_llm.bin
# 主机验证
gcc -O3 -o /tmp/verify_h2 firmware/host_verify/verify_h2.c -I firmware/esp32_llm_zh_v5 -lm
/tmp/verify_h2 firmware/model_v5/H2/model_llm.bin firmware/model_v5/H2/golden.txt  # 期望 PASS
```

---

## 陷阱与注意事项

1. **WSL 9p 文件系统**: 大文件/长脚本在 WSL 下可能触发 glibc 崩溃 (segfault), 用脚本文件方式运行, 避免 PowerShell 内联引号
2. **Windows vs WSL 路径**: 脚本内路径需双兼容 (`D:\...` 与 `/mnt/d/...`)
3. **CRLF 换行**: WSL 视角可能把 CRLF 视为 diff, 提交以 Windows 侧 git 为准
4. **RAFT 负样本**: 训练后模型对无关证据应拒答; 若无证据场景应不注入 system 提示 (盲引修复)
5. **量化**: SFT/RAFT 模型必须 int4 group=32 (group=128 会崩)
6. **模型输出必登记**: 见上文"模型输出规范"——CHANGELOG + MODELS.md 缺一不可
