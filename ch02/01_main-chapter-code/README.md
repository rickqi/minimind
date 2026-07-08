# 第 2 章:分词器 —— 从文本到 token id

> 📖 [中文导读](./ch02.md) | 📓 [主 notebook](./ch02.ipynb) | ⚡ [精简版](./tokenizer.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章从零拆解 BPE 算法、ByteLevel 预分词、special token 设计和 chat_template 渲染,解释 minimind 的 `vocab=6400` 如何在参数效率和压缩率之间取得平衡。

## 学习目标

- 手动实现极简 BPE,理解 merge rule 的训练和编码过程
- 说明 ByteLevel 为什么消灭 unknown token
- 列出 36 个 special token 的分组与用途
- 用 `apply_chat_template` 渲染 ChatML 格式对话
- 估算中英文文本的 token 数和压缩率

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch02.ipynb`](./ch02.ipynb) | 主 notebook(BPE 手动实现 + ByteLevel + special tokens + chat_template + 压缩率) |
| [`tokenizer.ipynb`](./tokenizer.ipynb) | 精简总结(BPE 一句话 + chat_template 示例 + special token 表) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(token 估算 / special token 原子性 / 扩展角色) |
| [`ch02.md`](./ch02.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `trainer/train_tokenizer.py:9-51`(tokenizer 训练:BPE + ByteLevel + special tokens)
- `model/tokenizer_config.json`(`chat_template` + added_tokens decoder)

---

← [教程总览](../../README.md) | → [第 3 章:配置](../ch03/01_main-chapter-code/README.md)
