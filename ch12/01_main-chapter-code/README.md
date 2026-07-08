# 第 12 章:对齐 I —— DPO:从偏好学习

> 📖 [中文导读](./ch12.md) | 📓 [主 notebook](./ch12.ipynb) | ⚡ [精简版](./dpo.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章介绍 DPO(Direct Preference Optimization)—— 消去 reward model,直接从偏好对(chosen/rejected)中学习对齐。从 Bradley-Terry 模型推导出闭式解,再逐行拆解 minimind 的损失函数、参考模型和训练配置。

## 学习目标

- 推导 DPO loss:Bradley-Terry → KL 闭式解 → 消去 reward model
- 手写 naive DPO loss,理解 minimind 的 batch 切分 compact 实现
- 解释参考模型角色:冻结 SFT 副本,防止策略偏离
- 说出 DPO vs SFT 的关键差异:lr=4e-8(低 250 倍)、beta=0.15、epochs=1

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch12.ipynb`](./ch12.ipynb) | 主 notebook(15 md + 7 code,教学载体) |
| [`dpo.ipynb`](./dpo.ipynb) | 精简总结(loss 公式 + 数据格式 + 配置表) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(β 作用 / 隐式 reward / 小差距) |
| [`ch12.md`](./ch12.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `trainer/train_dpo.py`(228 行):DPO 损失函数 + 训练循环
- `dataset/lm_dataset.py:122-192`:DPODataset + generate_loss_mask

## 核心公式

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \left[(\log \pi(y_w) - \log \pi(y_l)) - (\log \pi_{\text{ref}}(y_w) - \log \pi_{\text{ref}}(y_l))\right]\right)$$

## 核心数字

| 指标 | 值 |
|---|---|
| 学习率 | 4e-8(SFT 的 1/250) |
| beta | 0.15 |
| epochs | 1 |
| 显存 | SFT 的 ~2 倍(冻结 ref model) |

---

← [教程总览](../../README.md) | → [第 13 章](../ch13/01_main-chapter-code/README.md)
