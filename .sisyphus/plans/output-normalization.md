# 执行计划: 产物规范化 + MODELS.md 全面更新

> 生成日期: 2026-08-06
> 背景: 用户要求按 AGENTS.md §输出路径规划新规范调整现有产物命名/落位, 更新 MODELS.md 生成最新模型详细信息, 更新变更说明并提交推送, 评审方案。

---

## 一、GAP 分析结论 (已实测)

### 1. 文件命名/落位 (out/ 45 文件 + models/ 42 文件全量扫描)

**结论: 现有产物命名已基本符合新规范, 仅 1 处不符合:**

| 现有文件 | 问题 | 目标命名 |
|---|---|---|
| `out/mix_report.json` | 规范要求 `{pipeline}_report.json` 前缀 `medical_` | `out/medical_mix_report.json` |

- 其余 44 个 out/ 文件 + 42 个 models/ 文件**全部符合** `{阶段}_{架构}[_{变体}]_{dim}_ple.pth` / `_int4_g32.pth` / `_h{dim}_ple1.bin` / `_golden.{npz,txt}` 规范, **无需重命名/移动**。
- 检查点 `checkpoints/`、数据 `dataset/`、中间件(`rag_index.pkl`/`b2_cache.json`/`medical_jieba.txt`)落位均正确。

### 2. MODELS.md 登记缺口 (需补全)

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | §三 H2 系列表 | **H2 RAFT v4** (`full_sft_h2_raft_v4_384_ple.pth`, loss 2.70) 未入系列表, 仅在 §九自动登记 | 系列表补 v4 行 (含数据/Steps/Loss) |
| 2 | §五 部署表 | H2 行标识为 "H2 RAFT" 未标 v4; model_llm.bin 实际来自 RAFT v4 | 标注 v4, 更新说明 |
| 3 | §五 部署表 | H3/H3-raft model_llm.bin 声明 21.64MB, **实际 22.69MB** (已核对) | 修正为 22.69MB |
| 4 | §五 配套资产 | `out/rag_index.pkl` 描述 "jieba 医学检索索引" 正确, 保留 | 无 |
| 5 | §八 复现命令 | `wsl_train_h2_raft.sh` 注释 "输出 full_sft_h2_raft_v3" 已过时 (现为 v4) | 更新为 v4; `export_ple1.py --weight` 同步 v4 |
| 6 | §七 能力矩阵/结论 | H2 RAFT v3 → v4 (负样本+医学过滤, 盲引修复) | 更新引用版本 |

### 3. 部署产物核对 (esp32-ai firmware/model_v5)

| 目录 | 文件 | 实际大小 | 状态 |
|---|---|---|---|
| H1/ | model_llm.bin 6.31MB + model.bin 6.09MB + golden | ✅ 与登记一致 | 就绪 |
| H2/ | model_llm.bin 14.73MB + model.bin 14.07MB + golden | ✅ 与登记一致 (v4) | 就绪 |
| H3/ | model_llm.bin 22.69MB + golden | ⚠️ 大小声明过时 | 修正 21.64→22.69 |
| H3-raft/ | model_llm.bin 22.69MB + golden | ⚠️ 大小声明过时 | 修正 21.64→22.69 |

---

## 二、执行步骤

### Step 1: 重命名 `out/mix_report.json` → `out/medical_mix_report.json`
- `git mv` 不适用 (out/ gitignored), 直接文件重命名。
- 检查 `scripts/mix_medical.py` 是否写死输出名 → 若写死则同步改脚本 (保持可复现)。

### Step 2: 更新 `docs/MODELS.md` (核心工作量)
- §三 H2 系列表: 新增 **RAFT v4** 行:
  `| **RAFT v4** | out/full_sft_h2_raft_v4_384_ple.pth | 54.8MB | sft_medical_raft (8K, 负样本+医学过滤) | 1,000×3 | 3.20 → 2.70 |`
- §五 部署表: H2 行标注 **RAFT v4**; H3/H3-raft 大小修正为 22.69MB。
- §五 配套部署资产: 补充 `models/full_sft_h2_raft_v4_h384_ple1.bin` (14.73MB) 登记。
- §七 能力矩阵: "H2 RAFT v3" → "H2 RAFT v4"; 备注盲引修复验证结果。
- §八 复现命令: `wsl_train_h2_raft.sh` 注释 v3→v4; `export_ple1.py --weight full_sft_h2_raft_v4`。
- §九 自动登记: 保留 (register_model.py 已登记 v4)。
- 头部更新日期 → 2026-08-06。

### Step 3: 更新 `CHANGELOG.md`
- 在 [Unreleased] 追加条目: 产物规范化 (mix_report 重命名) + MODELS.md 全面更新 + H3 大小修正。
- 保留现有 H2 RAFT v4 登记。

### Step 4: 提交 + 推送 (minimind 仓库)
- 提交内容: `AGENTS.md` (分层知识库) + `scripts/AGENTS.md` + `trainer/AGENTS.md` + `docs/MODELS.md` + `CHANGELOG.md` + `scripts/mix_medical.py` (如改)。
- commit message: `docs: 分层 AGENTS.md 知识库 + 产物规范化与 MODELS.md 全面更新`
- push 到 origin master。

---

## 三、验证方式

1. 重命名后 `mix_medical.py` 再跑一次 dry-check 或确认脚本输出名已同步。
2. `docs/MODELS.md` 所有路径与磁盘实际文件逐一核对 (脚本比对)。
3. `CHANGELOG.md` [Unreleased] 区包含全部本次变更。
4. `git status` 干净, push 成功。

---

## 四、范围外 (明确不做)

- 不重训/不新导出任何模型。
- 不动 esp32-ai 侧文件 (部署产物已在正确位置)。
- 不修复 trainer PLE 后缀缺陷 (已在 trainer/AGENTS.md 记录, 另立任务)。
