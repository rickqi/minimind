# MiniMind 教程:从零理解一个 LLM 的每一行代码

> 参照 Sebastian Raschka《[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)》的教学风格,把 [minimind](https://github.com/jingyaogong/minimind) 这个"教学级 LLM 训练项目"拆成 **15 章可交互、中文为主**的教程。
>
> **核心理念**:不调用 `transformers` / `trl` / `peft` 的黑盒接口,**每一行核心代码都手写**,让你真正理解一个 LLM 从 0 到 1 的全过程。

---

## 这套教程是什么

- **被教的对象**:minimind —— 一个 ~64M 参数、Qwen3 架构对齐、训练成本约 ¥3 / 2 小时的完整 LLM 训练项目。
- **教学的风格**:Raschka 式 —— notebook 为主载体,markdown 解释 : 代码 ≈ 3 : 1,每个核心概念走 **naive → trainable → compact** 三段式,每个张量 shape 变化都注释。
- **覆盖的范围**:比 Raschka 原书多一整块 **「RL 对齐」**(DPO / PPO / GRPO / CISPO / Agent-RL),这是 minimind 相对原书的差异化价值。

## 章节目录(15 章 + 附录)

### 第一部分:模型半(ch1-7)—— 从零搭一个能跑的 minimind

| # | 标题 | 核心概念 |
|---|---|---|
| [1](./ch01/01_main-chapter-code/README.md) | 大图景:LLM 端到端是什么 | 先跑推理、看全貌 |
| 2 | 分词器:文本到 token id | BPE / ByteLevel / chat_template |
| 3 | 配置:每个超参数为何这样选 | dim / layers / GQA / SwiGLU 宽度 |
| 4 | 注意力原子:RMSNorm + RoPE + 单头 | 旋转位置编码 / GQA / QK-Norm |
| 5 | FFN 与 Block:SwiGLU 与残差流 | GLU 门控 / pre-norm 残差 |
| 6 | 模型组装:嵌入 / 堆叠 / 绑定权重 | forward 张量流 / CE loss |
| 7 | 生成:logits 怎么变成文本 | KV-cache / top-p / 流式 |

### 第二部分:训练半(ch8-11)—— 把权重练出来 + 部署

| # | 标题 | 核心概念 |
|---|---|---|
| 8 | 预训练:下一 token 预测循环 | AdamW / 余弦 LR / AMP / DDP |
| 9 | 监督微调(SFT):教模型扮演 assistant | **answer-only loss masking** |
| 10 | LoRA:参数高效微调从零实现 | 低秩分解 / monkey-patch / merge |
| 11 | 推理工程:API 服务 / 工具调用 | FastAPI SSE / `<tool_call>` 解析 |

### 第三部分:对齐 / RL 半(ch12-15)—— minimind 独有

| # | 标题 | 核心概念 |
|---|---|---|
| 12 | 对齐 I — DPO:从偏好学习 | chosen/rejected / 参考模型 / DPO loss |
| 13 | 对齐 II — RLAIF:PPO / GRPO / CISPO | **统一 PO 框架** / GAE / rollout |
| 14 | 智能体 RL:多轮工具使用作为轨迹 | 多轮 rollout / 延迟奖励 |
| 15 | 进阶:知识蒸馏与 MoE | CE+KL / top-1 路由 / aux_loss |

### 附录

- **appendix-A**:PyTorch 速成(前置)
- **appendix-B**:参考文献与延伸阅读
- **appendix-C**:习题解答索引
- **appendix-D**:训练循环增强(梯度裁剪等)

---

## 如何使用本教程

1. **按章顺序读**:章节间有严格依赖(`previous_chapters.py` carry-forward)。
2. **每章 4 件套**:
   - `ch##.ipynb` —— 主 notebook(教学载体,边读边跑)
   - `<topic>.ipynb` —— 精简总结(复习用)
   - `exercise-solutions.ipynb` —— 3 题 + 解答(独立)
   - `ch##.md` —— 中文导读(摘要 + 文件:行 引用)
3. **前置准备**:见 [`setup/README.md`](./setup/README.md)。
4. **需要 minimind 源码**:本教程引用真实文件:符号 + 行号,建议 clone 本仓库的 `master` 分支一份。

---

## 约定

- **语言**:中文为主,关键术语保留英文(RoPE / GQA / SwiGLU / KL 等)。
- **行号基准**:所有 `文件:行` 引用以本仓库 `master` 分支 commit `67f114a` 为准 (混合引用:函数/类名为主锚,`~行号` 为辅,标注 `@67f114a`)。
- **Fork 说明**:本仓库 (`rickqi/minimind`) 是上游 `jingyaogong/minimind` 的 fork。教程教的是**默认模式** (`use_ple=False`),行为与上游原版完全一致;fork 扩展 (PLE 架构) 以 `🔧 Fork 扩展` sidebar 标注。
- **代码注释**:每个张量 shape 变化都行内注释 `# Shape: (b, T, H) -> (b, T, n_h, d_h)`。

> "亲手用乐高搭一架飞机,远比坐头等舱更令人兴奋。" —— minimind 项目哲学
