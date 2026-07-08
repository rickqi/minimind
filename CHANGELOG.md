# Changelog

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
