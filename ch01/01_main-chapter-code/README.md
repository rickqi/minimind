# 第 1 章:大图景 —— LLM 端到端是什么

> 📖 [中文导读](./ch01.md) | 📓 [主 notebook](./ch01.ipynb) | ⚡ [精简版](./big-picture.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章是整个教程的入口。我们**不写模型代码**,而是先把 minimind 跑起来,看到一个 64M 参数的 LLM 生成文字,然后拆解推理过程,建立对后续 14 章的全局认知。

## 学习目标

- 理解 LLM 的 6 个生命周期阶段
- 亲手加载并运行 minimind 推理
- 拆解推理的 4 个步骤(tokenizer → model → sampling → chat_template)
- 看懂 15 章路线图

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch01.ipynb`](./ch01.ipynb) | 主 notebook(教学载体,边读边跑) |
| [`big-picture.ipynb`](./big-picture.ipynb) | 精简总结(快速复习) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答 |
| [`ch01.md`](./ch01.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `eval_llm.py`(98 行,全文)
- `model/model_minimind.py:~314-345` (@67f114a)(`generate` 方法,第 7 章详解)

---

← [教程总览](../../README.md) | → [第 2 章:分词器](../ch02/01_main-chapter-code/README.md)
