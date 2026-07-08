# 第 4 章:注意力原子 —— RMSNorm + RoPE + 单头注意力

> 📖 [中文导读](./ch04.md) | 📓 [主 notebook](./ch04.ipynb) | ⚡ [精简版](./attention.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章是全教程的**核心章节**。我们从最朴素的 naive self-attention 出发,逐步加入 causal mask、multi-head、RoPE、GQA、QK-Norm、KV cache,最终拼出 minimind 的 44 行 `Attention` 类。

## 学习目标

- 从零写出 naive → compact 的 self-attention
- 画出 multi-head 的完整 shape 流程并注释每一步
- 理解 RoPE 旋转、GQA 共享、QK-Norm 稳定化的原理
- 逐行读懂 `model_minimind.py:91-134`

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch04.ipynb`](./ch04.ipynb) | 主 notebook(naive→compact 全流程,21 md + 12 code) |
| [`attention.ipynb`](./attention.ipynb) | 精简总结(shape 流程图 + 速记公式) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(手算/expand/QK-Norm) |
| [`ch04.md`](./ch04.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `model/model_minimind.py:50-60`(`RMSNorm`)
- `model/model_minimind.py:62-89`(`precompute_freqs_cis` / `apply_rotary_pos_emb` / `repeat_kv`)
- `model/model_minimind.py:91-134`(`Attention` 类,全章拆解对象)

---

← [教程总览](../../README.md) | → [第 5 章:FFN 与 Block](../ch05/01_main-chapter-code/README.md)
