# 第 7 章:生成 —— logits 怎么变成文本

> 📖 [中文导读](./ch07.md) | 📓 [主 notebook](./ch07.ipynb) | ⚡ [精简版](./generation.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章从零拆解 minimind 手写的 `generate` 方法(32 行):greedy → temperature → top-k → top-p → repetition penalty → KV cache,遵循 naive→compact 路径逐步构建完整的自回归生成循环。

## 学习目标

- 手写朴素 greedy decoding 循环(无 KV cache),理解其确定性和重复问题
- 说明 temperature/top-k/top-p 如何控制采样分布的锐度和截断
- 实现 repetition penalty 的正负分支处理
- 追踪 KV cache shape 变化,解释 $O(T^2) \to O(T)$ 加速
- 说出 `generate` 中采样策略的执行顺序和 minimind 为何覆盖 GenerationMixin

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch07.ipynb`](./ch07.ipynb) | 主 notebook(greedy→sampling→KV cache→完整 generate 拆解) |
| [`generation.ipynb`](./generation.ipynb) | 精简总结(greedy/sampling 一行代码 + KV cache 图 + 参数表) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(greedy 实现 / KV cache 复杂度 / top_k vs top_p 顺序) |
| [`ch07.md`](./ch07.md) | 中文导读(摘要 + 行号引用 + 术语表 + 执行顺序图) |

## 对应 minimind 源码

- `model/model_minimind.py:256-288`(手写 `generate`:KV cache + 采样策略 + EOS + streamer)
- `model/model_minimind.py:234`(`GenerationMixin` 继承与覆盖)

---

← [教程总览](../../README.md) | → [第 8 章](../ch08/01_main-chapter-code/README.md)
