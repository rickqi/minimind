# 第 3 章:配置 —— 每个超参数为何这样选

> 📖 [中文导读](./ch03.md) | 📓 [主 notebook](./ch03.ipynb) | ⚡ [精简版](./config.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章逐行拆解 `MiniMindConfig`(`model/model_minimind.py:10-45`)的每一个超参数,解释为什么是 `hidden_size=768`、`intermediate_size=2432`、`rope_theta=1e6`,并用代码验证 64M 参数量的精确来源。

## 学习目标

- 说出 `MiniMindConfig` 的 16 个核心字段及其取值理由
- 手算 GQA 省了多少 KV cache 参数
- 推导 `intermediate_size = ⌈768·π/64⌉·64 = 2432` 的来源
- 计算 dense(64M)和 MoE(198M total / 64M active)的精确参数量
- 理解 minimind 与 Qwen3 的权重级兼容

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch03.ipynb`](./ch03.ipynb) | 主 notebook(逐行拆解 16 个超参数 + 参数量计算) |
| [`config.ipynb`](./config.ipynb) | 精简总结(配置一览表 + 参数量公式) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(含代码验算) |
| [`ch03.md`](./ch03.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `model/model_minimind.py:10-45`(`MiniMindConfig` 类,全部超参数定义)
- `scripts/convert_model.py:43-71`(MiniMindConfig → Qwen3Config 字段映射)

---

← [教程总览](../../README.md) | → [第 4 章:注意力](../ch04/01_main-chapter-code/README.md)
