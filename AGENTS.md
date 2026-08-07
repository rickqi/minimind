# AGENTS.md

> 本文件为 OpenCode/Claude 等 AI 代理提供本仓库的关键上下文, 避免踩坑。
> 语言: 中文 (与仓库内 README.md / docs/*.md 一致)
>
> **分层知识库**: `scripts/AGENTS.md` (驱动层约定) · `trainer/AGENTS.md` (训练引擎契约)
> 修改这两目录代码前必读对应子文件。

---

## 仓库定位

**MiniMind** — 本项目的目的: **处理训练数据 → 训练模型 → 发布导出指定模型相关数据和文件**。
完整链路 (PLE 架构, Per-Layer Embedding): 模型训练 (PyTorch 原生, WSL-GPU)、医疗数据管线、量化导出、ESP32 部署产物生成与主机验证。

**项目边界 (明确)**:
- ✅ **负责**: 训练数据处理/清洗/构建 → 模型训练 (pretrain/SFT/DPO/RAFT) → 量化导出 (int4/int8) → 部署产物生成 (PLE1 + golden) → 主机验证 (verify PASS) → 产物登记发布 (MODELS.md + CHANGELOG)
- ❌ **不负责**: ESP32 实机烧录/实机验证 (COM 口操作不属于本项目职责), 旧固件 (V1-V3) 维护 (已冻结, 只维护 V5)
- 📦 **交付物**: 训练权重 (out/)、部署二进制 (models/ + esp32-ai/)、数据管线报告、模型清单文档

- 训练环境: **WSL Ubuntu-22.04 + RTX 5080** (torch 2.9.1+cu128), 训练脚本在 `scripts/wsl_*.sh`
- 模型架构: H1 (d256/l6/p96) / H2 (d384/l8/p128) / H3 (d512/l8/p128), 词表 6400 (BPE)
- 医疗数据: 5 条管线 (A/B1/B2/RAFT/混合), 详见 `docs/MEDICAL_TRAINING.md`
- 模型清单: 全部训练/部署产物登记在 `docs/MODELS.md`

### 🏗️ 两类训练模式(重要)

本项目训练代码分两类, 处理时必须区分:

| 类型 | 说明 | 数据 | 代码基线 |
|---|---|---|---|
| **1. 默认自带模式** | 项目 fork 自上游 minimind 的原生训练链路 (pretrain/SFT/DPO/GRPO/PPO/Agent/LoRA/蒸馏), **是基础** | 官方数据 (pretrain_t2t_mini / sft_t2t_mini / dpo) | `trainer/*.py` + `model/*.py` 上游原版 |
| **2. 自有数据分支尝试** | 基于自有医疗数据在默认模式上的扩展 (PLE 架构 / RAFT / RAG / 医疗管线) | 医疗数据 (5 管线) | PLE 修改版 + `scripts/*` + `docs/*` |

**规则 (硬性)**:
- **默认模式代码必须保留** — 上游原版 trainer/model 不被破坏
- **所有新尝试代码避免修改默认训练代码** — 新增用新文件 (scripts/ 等) 或清晰标注的 PLE 分支 (use_ple 参数隔离, 默认 False 保持上游行为)
- **本项目默认还会进一步更新** (上游 minimind 持续迭代) — 需保持可拉取上游

### 🌿 分支与冲突策略(重要)

- **master 分支** = 当前各类尝试的集合 (PLE/RAFT/RAG/医疗)
- **冲突评估 (2026-08-07 实测)**: 本地深改的上游文件 (`model/model_minimind.py`, `trainer/trainer_utils.py`, `train_pretrain/full_sft/dpo.py`) **拉取上游必然 merge conflict**; 本地新增文件 (scripts/ docs/ dataset/ AGENTS.md) 零冲突
- **开发规范**: 新尝试尽量新增文件或隔离参数 (use_ple 默认 False), 减少对上游原版的侵入

### 📥 上游同步流程(默认版本管理要求, 2026-08-07 已验证)

**必须遵守** — 拉取上游 minimind 更新时的唯一合法流程 (已验证于 `fe19dfa`, 零冲突):

```bash
# 1. 检查上游是否有新提交 (upstream remote 已配置)
git fetch upstream
git log --oneline master..upstream/master    # 看上游领先多少

# 2. 创建合并分支 (禁止直接 git pull 到 master)
git checkout -b upstream-merge

# 3. 合并上游 (冲突在此分支解决, master 不受影响)
git merge upstream/master --no-edit

# 4. 解决冲突 (若出现):
#    - 本地 PLE 相关文件冲突 → 保留本地 use_ple 隔离逻辑 + 合并上游新逻辑
#    - 纯文档/新增文件 → 通常 auto-merge 干净
# 5. 验证 (必须):
#    - python -c "import py_compile; [py_compile.compile(f, doraise=True) for f in ['model/model_minimind.py','trainer/trainer_utils.py']]"
#    - 加载 PLE 权重: MiniMindForCausalLM(cfg) + load_state_dict → missing=0 unexpected=0
#    - 确认 use_ple=True 行为未变

# 6. 合回 master + 推送
git checkout master
git merge upstream-merge --no-edit
git push origin master
git branch -d upstream-merge
```

**规则**:
- **禁止直接 `git pull upstream` 到 master** — 必用 upstream-merge 分支
- 合并前必跑 PLE 权重加载验证 (missing=0 unexpected=0)
- 合并后更新 CHANGELOG 记录上游同步点
- 上游 remote: `git@github.com:jingyaogong/minimind.git` (已配置为 `upstream`)

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
    --data "sft_medical_raft (8K, 负样本)" --loss "2.70" --verify "PASS diff 0.00001"
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
- **ESP32 flash 容量红线**: 16MB 总 flash, model 分区 14.5MB → H1 (6.31MB) / H2 (14.73MB) 可烧录 (H2 余量极小), **H3 (22.69MB) 超限, 仅 PC/树莓派部署**

---

## 🔍 RAG 体系 (医学精准问答核心, 跨两仓库)

**精准问答 = RAG + RAFT 复述** (无 RAG 时 H1/H2 均退化/幻觉)。三条独立索引链, 各有用途:

| 索引链 | 位置 | docs | 用途 |
|---|---|---|---|
| **PC jieba** | `out/rag_index.pkl` | 5,891 (医学过滤) | `rag_medical.py` 评估/PC 注入 |
| **SD 三文件** | `esp32-ai/data_v4/sd_rag/{index,docs,meta}.bin` | 10,999 (v3 医学成品) | `esp32_llm_v5_idf` 离线 RAG (拷入 `/sdcard/rag/`) |
| **flash RAG1** | `esp32-ai/data_v4/kb/index.bin` | 10,999 | 仅 v2/v4 旧固件; V5 设备端 RAG 为死代码 |

**关键约束** (详见 `scripts/AGENTS.md` 陷阱 6-10 + `esp32-ai/docs/RAG_INDEX_ANALYSIS_20260806.md`):
- **索引新鲜度**: `rag_index.pkl` 是 build 时快照, 词典/过滤变更后**必须重建** (`python scripts/rag_medical.py build`), 否则医学术语检索失败
- **部署=评估一致性**: `send_prompt_rag.py` (部署) 与评估脚本必须**同样加载医学词典 + med_only 过滤**
- **实测结论**: 修复后 jieba 10 查询 Top-1 90% > SD 单字 80%; SD 对肝豆状核/白疕有字符级结构性误配
- **SD 索引 v3**: 用 `format_data.jsonl` (11K 医学成品) 构建, 精度>召回; 产物在 esp32-ai 仓库 git 追踪, 重跑会覆盖磁盘文件

---

## 📦 输出路径规划 (交付物规范, 所有产物必须落位)

**命名统一**: `{阶段}_{架构}[_{变体}]_{dim}_ple.pth` (阶段: pretrain/full_sft/dpo/grpo/ppo/agent; 架构: h1/h2/h3; 变体: med/pure/raft/raft_v4/mixed/mixed_raft)

| 产物类型 | 存放目录 | 命名规则 | 示例 | 追踪方式 |
|---|---|---|---|---|
| 训练权重 (fp16) | `out/` | `{阶段}_{架构}[_{变体}]_{dim}_ple.pth` | `full_sft_h2_raft_v4_384_ple.pth` | MODELS.md 系列表 |
| 训练日志 | `out/` | `{权重名}.log` (与权重同名) | `full_sft_h2_raft_v4.log` | CHANGELOG Steps/Loss 来源 |
| 数据管线报告 | `out/` | `{pipeline}_report.json` | `medical_sft_raft_report.json` | 数据质量审计 |
| 中间件 (RAG/缓存) | `out/` | `rag_index.pkl` / `b2_cache.json` / `medical_jieba.txt` | — | 可重建, 不登记 |
| SD 卡 RAG 索引 | `esp32-ai/data_v4/sd_rag/` | `{index,docs,meta}.bin` (三文件) | 10,999 docs (11K 医学成品) | esp32-ai 侧提交 + MODELS.md SD 小节 |
| 断点续训包 | `checkpoints/` | `{权重名}_{dim}_ple*_resume.pth` | — | 训练恢复用 |
| int4 量化权重 | `models/` | `{权重名}_{dim}_int4_g32.pth` | `full_sft_h2_raft_v4_384_int4_g32.pth` | MODELS.md 量化小节 |
| PLE1 扁平二进制 | `models/` | `{权重名}_h{dim}_ple1.bin` | `full_sft_h2_raft_v4_h384_ple1.bin` | MODELS.md 部署区 |
| golden 验证对 | `models/` | `{权重名}_h{dim}_golden.{npz,txt}` | — | 随 ple1 同批次生成 |
| ESP32 部署产物 | `../esp32-ai/firmware/model_v5/{H1,H2,H3}/` | `model_llm.bin` + `golden.txt` | — | esp32-ai 侧提交 |
| 训练数据 | `dataset/` | `{来源}_{类型}.jsonl` | `sft_medical_raft.jsonl` | MODELS.md 数据资产节 |

**产出即登记**: 任何新权重/量化/部署产物生成后, 用 `register_model.py` 登记或手动按 §模型输出规范更新 CHANGELOG.md + MODELS.md, 缺一不可。

---

## 精确命令速查

### 医疗数据管线
```bash
python scripts/build_medical_pretrain.py                    # 管线A: 预训练语料
python scripts/build_medical_sft_b1.py                      # 管线B1: 直接转换
python scripts/build_medical_sft_b2.py --api-key $KEY       # 管线B2: V4 Flash 合成
python scripts/build_medical_sft_pure.py                    # 纯医学: B1过滤 + B2 合并
python scripts/build_medical_raft.py --no-evidence-ratio 0.3 --negative-ratio 0.15 --med-only
python scripts/mix_medical.py                               # 混合 (1:2 / 1:3)
```

### 训练 (WSL GPU)
```bash
bash scripts/wsl_train_h2_raft.sh        # H2 RAFT 微调 (输出 full_sft_h2_raft_v4)
bash scripts/wsl_train_h1_raft.sh        # H1 RAFT v4 微调 (输出 full_sft_h1_raft_v4)
bash scripts/wsl_train_h3_mixed.sh       # H3 混合从零预训练
bash scripts/wsl_sft_h3_mixed.sh         # H3 混合 SFT
bash scripts/wsl_train_h3_mixed_raft.sh  # H3 混合+RAFT (组合验证)
# 完整清单: wsl_train_h{1,2,3}.sh / wsl_sft_h{1,2,3}.sh / wsl_dpo_h{1,2,3}.sh
```
- 训练脚本统一 `cd trainer` + `python3 -u train_*.py`, 通过 `--from_weight` 续训
- 输出到 `out/{save_weight}_{hidden_size}_ple.pth`, 权重带 `_ple` 后缀 (不与 dense/moe 冲突)
- 用 `--use_ple 1 --ple_dim N` 启用 PLE 架构

### 评估
```bash
bash scripts/wsl_eval_raft_v4.sh         # RAFT v4 问答 (负样本/盲引验证)
bash scripts/wsl_eval_h1_raft_v4.sh      # H1 RAFT v4 问答 (调用 scripts/eval_h1_raft_v4.py)
bash scripts/wsl_eval_rag_compare.sh     # H1/H2 RAG vs 无RAG
bash scripts/wsl_eval_4models.sh         # 4 模型横向对比
# 完整清单: 16 个 wsl_eval_*.sh, 详见 scripts/AGENTS.md
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
- **GQA→MHA**: H1/H2 是 kv_heads=4, ESP32 llm_v5.h 是标准 MHA, 导出时 `repeat_interleave(2)` 自动扩展
- **golden 必须来自反量化模型** (int4 模拟), 否则 verify 测的是量化误差而非转换正确性

---

## 陷阱与注意事项

1. **WSL 9p 文件系统**: 大文件/长脚本在 WSL 下可能触发 glibc 崩溃 (segfault / double free)。
   规避: 用脚本文件方式运行, 避免 PowerShell 内联引号; 清理 `__pycache__` 后再跑。
2. **Windows vs WSL 路径**: 脚本内路径需双兼容 (`D:\...` 与 `/mnt/d/...`); KB 等外部数据用 WSL 路径 `/mnt/d/...`。
3. **CRLF 换行**: WSL 视角可能把 CRLF 视为 diff (31 文件全变红是误报), 提交以 Windows 侧 git 为准。
4. **RAFT 负样本**: 训练后模型对无关证据应拒答; 若无证据场景应不注入 system 提示 (盲引修复)。
5. **量化**: SFT/RAFT 模型必须 int4 group=32 (group=128 会崩, group=16 过拟合)。
6. **jieba 超长文本**: 去重/分词时对 >5000 字符文本截断, 否则 DAG 溢出崩溃。
7. **模型输出必登记**: 见上文"模型输出规范"——CHANGELOG + MODELS.md 缺一不可。
8. **脚本注释可能过期**: 训练脚本顶注释描述的是初始版本, 实际参数以脚本内容为准。
