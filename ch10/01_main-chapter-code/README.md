# 第 10 章:LoRA —— 从零实现参数高效微调

> 📖 [中文导读](./ch10.md) | 📓 [主 notebook](./ch10.ipynb) | ⚡ [精简版](./big-picture.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章介绍 LoRA(Low-Rank Adaptation)—— 冻结原始权重,只训练一个极小的低秩「补丁」。采用 **naive → compact** 教学路径:先手写简单 LoRA layer 理解原理,再展示 minimind 的 monkey-patch 工程实现。

## 学习目标

- 理解 LoRA 数学原理:$W' = W + BA$,参数从 $d^2$ 降到 $2rd$
- 手写 naive LoRA layer 并理解零初始化技巧($B = \mathbf{0}$)
- 读懂 minimind 的 `apply_lora`:monkey-patch + 方形过滤
- 说出 LoRA 训练与全参 SFT 的关键差异(冻结、lr=1e-4、compile 不兼容)
- 演示 save → load → merge 完整流程

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch10.ipynb`](./ch10.ipynb) | 主 notebook(教学载体,边读边跑) |
| [`big-picture.ipynb`](./big-picture.ipynb) | 精简总结(快速复习) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答 |
| [`ch10.md`](./ch10.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `model/model_lora.py`(65 行,全文):LoRA 模块 + apply / save / load / merge
- `trainer/train_lora.py`(186 行):冻结非 LoRA 参数,只训 LoRA

## 核心数字

| 指标 | 值 |
|---|---|
| LoRA 参数量 | 393,216(0.39M) |
| LoRA 占比 | 0.61% |
| 保存文件大小 | ~0.8 MB |
| 学习率 | 1e-4(SFT 的 10×) |
| 被挂 LoRA 的模块 | q_proj + o_proj × 8 层 = 16 个 |

---

← [教程总览](../../README.md) | → [第 11 章:推理工程](../ch11/01_main-chapter-code/README.md)
