# 第 5 章:FFN 与 Block —— SwiGLU 与残差流

> 📖 [中文导读](./ch05.md) | 📓 [主 notebook](./ch05.ipynb) | ⚡ [精简版](./ffn.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章从朴素 FFN 演进到 SwiGLU(`model/model_minimind.py:136-146`),扩展到 MoE(148-176),最后组装成 MiniMindBlock(178-194)。涵盖 GLU 门控、SiLU 激活、`intermediate_size=2432` 的 π 公式、MoE 路由与 aux_loss、Pre-Norm 残差连接。

## 学习目标

- 手写 SwiGLU 前向:`down_proj(silu(gate_proj(x)) * up_proj(x))`
- 解释 SiLU 优于 ReLU 的原因(负区有梯度)
- 推导 `intermediate_size = ⌈768·π/64⌉·64 = 2432`
- 描述 MoE 的 top-1 路由和 aux_loss 负载均衡
- 画出 MiniMindBlock 的双残差数据流

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch05.ipynb`](./ch05.ipynb) | 主 notebook(FFN 演进 + MoE + Block,15 md + 9 code) |
| [`ffn.ipynb`](./ffn.ipynb) | 精简总结(SwiGLU 代码 + shape 流动 + Block 结构图) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答(swap 激活/手算参数/aux_loss) |
| [`ch05.md`](./ch05.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `model/model_minimind.py:136-146`(`FeedForward` —— SwiGLU)
- `model/model_minimind.py:148-176`(`MOEFeedForward` —— MoE 路由)
- `model/model_minimind.py:178-194`(`MiniMindBlock` —— 双残差 Block)

---

← [第 4 章:注意力](../ch04/01_main-chapter-code/README.md) | [教程总览](../../README.md) | → [第 6 章:RMSNorm](../ch06/01_main-chapter-code/README.md)
