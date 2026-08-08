# 第 6 章:模型组装 —— 嵌入 → 堆叠 Block → lm_head → 绑定权重

> 📖 [中文导读](./ch06.md) | 📓 [主 notebook](./ch06.ipynb) | ⚡ [精简版](./big-picture.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

前 3 章我们拆解了 minimind 的零件(Config、RMSNorm、RoPE、Attention、FeedForward、Block)。本章是**总装车间** —— 把这些零件拼成一台完整的引擎。

核心是**完整 forward pass 的 tensor shape 追踪**:从 `input_ids (1, 4)` 到 `logits (1, 4, 6400)` 到 `loss`,逐步打印每一步的形状变化。

## 学习目标

- 解释 `nn.Embedding` 如何把 token id 查表变成向量
- 说出 `start_pos` 的计算逻辑和 RoPE 窗口切片原理
- 理解 `register_buffer(persistent=False)` 和权重绑定的机制
- 从 input_ids 到 loss 追踪完整 forward pass 中每一步的 tensor shape
- 解释 CE loss 的 shift-by-1 对齐和 `ignore_index=-100` 的作用
- 手动拆解 64M 模型的参数量

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch06.ipynb`](./ch06.ipynb) | 主 notebook(10 节,逐步追踪 forward pass 的 shape 变化) |
| [`big-picture.ipynb`](./big-picture.ipynb) | 精简总结(流水线图 + 关键代码 + 参数量表) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(含 logits_to_keep 在 RL 中的优化原理) |
| [`ch06.md`](./ch06.md) | 中文导读(摘要 + 文件:行引用 + 术语表 + 参数量分布) |

## 对应 minimind 源码

- `model/model_minimind.py:210-264`(`MiniMindModel`:backbone)
- `model/model_minimind.py:266-310`(`MiniMindForCausalLM`:lm_head + loss + tied weights)

> `generate`(314-345)是第 7 章的主题,本章不教。

## 章节结构

| 节 | 主题 | 关键 shape |
|---|---|---|
| 6.1 | Embedding 层 | `(1, 4) → (1, 4, 768)` |
| 6.2 | 堆叠 N 个 Block + RoPE buffer | `(1, 4, 768)` 不变 |
| 6.3 | forward 的 start_pos 计算 | KV cache `(1,4,4,96) → (1,5,4,96)` |
| 6.4 | 最终 RMSNorm | `(1, 4, 768)` 不变 |
| 6.5 | lm_head | `(1, 4, 768) → (1, 4, 6400)` |
| 6.6 | 权重绑定(Tied Embeddings) | 省 7.7% 参数 |
| 6.7 | 完整 forward pass | 全流程 shape 追踪 |
| 6.8 | CE Loss 与 -100 masking | shift by 1 + ignore_index |
| 6.9 | MoE aux_loss 累加 | 密集模型=0,MoE 训练≠0 |
| 6.10 | 参数量验证 | 63,912,192 ≈ 63.9M |

---

← [教程总览](../../README.md) | → [第 7 章:生成 —— logits 怎么变成文本](../ch07/01_main-chapter-code/README.md)
