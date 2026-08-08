# Changelog

## 2026-08-08: Align to master 67f114a (PLE fork extension)

### Changed
- **行号基准重新锁定**: commit `512eed0` → master `67f114a` (README / ANALYSIS / setup / 全部章节)
- **clone 目标**: `jingyaogong/minimind` (上游) → `rickqi/minimind` (本 fork);默认模式 `use_ple=False` 行为与上游一致
- **混合引用格式**: 全部 `文件:行` 改为 `SymbolName — file.py:~NN (@67f114a)` (函数名主锚 + 近似行号 + commit 标注)
- **ANALYSIS.md**: §2a Config 表补 `use_ple`/`ple_dim` + 修正全部字段行号;§2b 构件表行号 +5 级联;§2c Model/CausalLM/generate 行号 +14~+57;§7 章节映射表全面更新;§4 共享骨架补 `_model_suffix`;`model_minimind.py` 288→344 行;`eval_llm.py` 94→98 行

### Added
- **PLE sidebar** (🔧 Fork 扩展): ch03 (Config `use_ple`/`ple_dim` 字段) + ch06 (`MiniMindModel.forward` PLE 注入段 + enumerate 循环桥接说明)
- **ch06 forward 重写**: `zip(self.layers, past_kv)` → master 的 `enumerate(self.layers)` + `ple[:,:,i]` 按层切片;附桥接注 (ple=None 时等价于更简洁的 zip)
- **ch08 `_model_suffix()` 教学**: checkpoint 后缀统一逻辑 (`_ple`/`_moe`/空),默认 Dense 路径行为不变
- **ch13 PPO 3 处行为变更**:
  - ① `mb_resp_logp` 移入 `autocast_ctx` — fp16/bf16 logits 直接 log_softmax 的数值偏差 bugfix
  - ② `ppo_train_epoch` 移除 `use_sglang` 形参 — sglang 路由下沉到 rollout_engine
  - ③ `--debug_log_ratio` 诊断工具 — 核查首轮 ratio≈1 假设 (默认关闭)

### Updated (line numbers only)
- ch04 Attention: model_minimind.py 全部行号 +5 (内联教学代码逻辑未变)
- ch05 FFN/Block: 行号 +5~+14;MiniMindBlock.forward 签名补 `ple=None` 注释
- ch07 Generate: 行号 +57~+58
- ch01/ch15: eval_llm.py +6 行号;MOEFeedForward 148-176→153-181;"288 行"→"344 行"
- ch09/10/12/14: checkpoint 后缀片段 `_model_suffix` 说明
- ch04/07 exercise-solutions: 行号同步

### Verified
- 全部 .ipynb JSON 合法 (0 非法)
- 零残留旧引用 (512eed0 / 94 行 / 288 行 / 旧行号区间 / 旧 clone URL)
- 42 files changed, +762/−302

---

## 2026-07-08: Initial Release

### Added
- 15-chapter tutorial covering minimind's full LLM pipeline (model → training → alignment/RL)
- 45 Jupyter notebooks (main + summary + exercise-solutions per chapter)
- 15 Chinese chapter guides (ch##.md) with minimind file:line references
- 4 appendices: PyTorch crash course, references, exercise index, training enhancements
- Shared notebook generator (`_tools/gen_notebook.py`) and validator (`_tools/validate.py`)
- Codebase analysis report (`ANALYSIS.md`) mapping all 22 minimind source files
- Evaluation report verifying end-to-end executability (build → train → save → load → inference)

### Tutorial Structure
- **ch01-07 (Model Half)**: Big picture → tokenizer → config → attention (RMSNorm+RoPE+GQA) → FFN (SwiGLU+MoE) → model assembly → generation
- **ch08-11 (Training Half)**: Pretraining loop → SFT (answer-only loss masking) → LoRA → inference engineering
- **ch12-15 (Alignment/RL Half)**: DPO → PPO/GRPO/CISPO unified framework → Agent-RL → distillation + MoE

### Quality Metrics
- 60 validation checks passed, 0 errors
- 45 exercises with folded solutions
- End-to-end pipeline verified: 63.9M model → loss decrease → checkpoint → inference
