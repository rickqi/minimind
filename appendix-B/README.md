# 附录 B:参考文献与延伸阅读

## 核心论文

### Transformer 架构
- **Attention Is All You Need** (Vaswani et al., 2017) —— Transformer 原始论文
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) —— RoPE
- **GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023) —— GQA
- **GLU Variants Improve Transformer** (Shazeer, 2020) —— SwiGLU
- **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019) —— RMSNorm

### 预训练
- **GPT-3: Language Models are Few-Shot Learners** (Brown et al., 2020)
- **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
- **Qwen3 Technical Report** —— minimind 架构对齐的目标

### 微调
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)

### 对齐 / RL
- **RLHF: Training language models to follow instructions with human feedback** (OpenAI, 2022)
- **DPO: Direct Preference Optimization** (Rafailov et al., 2023)
- **PPO: Proximal Policy Optimization Algorithms** (Schulman et al., 2017)
- **GRPO: DeepSeekMath** (Shao et al., 2024)
- **CISPO** —— minimind 的默认 RL 算法

### MoE
- **Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer** (Shazeer et al., 2017)
- **Mixtral of Experts** (Mistral AI, 2024)

### 蒸馏
- **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)

### 长文本
- **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023)

## 参考项目

- **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)** —— Sebastian Raschka 的「从零构建 LLM」书籍配套代码。本教程的教学风格直接参考此项目。
- **[minimind](https://github.com/jingyaogong/minimind)** —— 本教程的教学对象。
- **[Qwen3](https://github.com/QwenLM/Qwen3)** —— minimind 架构对齐的工业级实现。

## 延伸阅读

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) —— Jay Alammar 的经典可视化教程
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) —— Harvard NLP 的逐行注释实现
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajvE) —— Andrej Karpathy 的 BPE 讲解
