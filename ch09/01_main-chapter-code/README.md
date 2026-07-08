# 第 9 章:监督微调(SFT)—— 教模型扮演 assistant

> 📖 [中文导读](./ch09.md) | 📓 [主 notebook](./ch09.ipynb) | ⚡ [精简总结](./sft.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

预训练模型只会「续写文本」,不会「回答问题」。**监督微调(SFT)** 用对话数据教会模型扮演 assistant —— 给定用户的提问,生成有意义的回复。

核心是 **answer-only loss masking**:只让模型从 assistant 的回复 token 学习,忽略 prompt(user/system)部分。

## 学习目标

- 解释预训练和 SFT 的本质区别(学语言 vs 学对话)
- 说出 SFT 数据格式(`{"conversations":[{role, content}]}`)和 chat_template 渲染过程
- **手动实现 `generate_labels`**:定位 assistant span,设置 loss masking
- 可视化哪些 token 参与 loss(绿)vs 被忽略(灰)
- 对比 SFT vs Pretrain 的关键差异(学习率低 50 倍、label 仅 assistant、max_seq_len 更长)
- 解释 catastrophic forgetting 与为什么 SFT 学习率必须很低

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch09.ipynb`](./ch09.ipynb) | 主 notebook(8 节,含 loss masking 可视化 + 多轮对话演示) |
| [`sft.ipynb`](./sft.ipynb) | 精简总结(loss masking 可视化 + SFT vs Pretrain 对照表) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(含 catastrophic forgetting 模拟) |
| [`ch09.md`](./ch09.md) | 中文导读(摘要 + 文件:行引用 + 术语表 + 对照表) |

## 对应 minimind 源码

- `dataset/lm_dataset.py:58-119`(`SFTDataset` + `generate_labels`)
- `dataset/lm_dataset.py:9-35`(`pre_processing_chat` + `post_processing_chat`)
- `trainer/train_full_sft.py`(训练脚本:lr=1e-5, from_weight=pretrain)
- `eval_llm.py`(`--weight full_sft` 对比 pretrain)

## 章节结构

| 节 | 主题 | 关键点 |
|---|---|---|
| 9.1 | 为什么预训练后还要 SFT | 预训练=续写,SFT=对话 |
| 9.2 | SFT 数据格式 | `{"conversations":[{role, content}]}` |
| 9.3 | chat_template 渲染 | `apply_chat_template → <|im_start|>role\n...<|im_end|>` |
| 9.4 | ⭐ answer-only loss masking | `generate_labels`:assistant span → label=token_id,其余 -100 |
| 9.5 | pre/post processing | 20% 加 system prompt,80% 去空 think |
| 9.6 | SFT vs Pretrain 差异 | lr 1e-5 vs 5e-4,label 仅 assistant vs 全部 |
| 9.7 | 训练命令与结果 | `python train_full_sft.py` → `full_sft_768.pth` |
| 9.8 | 验证 | `eval_llm.py --weight full_sft` 对比 pretrain 输出 |

---

← [教程总览](../../README.md) | → [第 10 章:DPO 直接偏好优化](../ch10/01_main-chapter-code/README.md)
