# 第 11 章:推理工程 —— API 服务 / 流式 / 工具调用

> 📖 [中文导读](./ch11.md) | 📓 [主 notebook](./ch11.ipynb) | ⚡ [精简版](./serving.ipynb) | ✏️ [习题与解答](./exercise-solutions.ipynb)

本章解决最后一个工程问题:**如何把训练好的模型变成可用的服务**。用 FastAPI 搭建 OpenAI 兼容的 `/v1/chat/completions` 端点,实现 SSE 流式输出,解析 `<think>` 和 `<tool_call>`,最后转换为 HF transformers 格式部署到 vllm / ollama。

## 学习目标

- 用 FastAPI 搭建 OpenAI 兼容 API 服务
- 理解 SSE 流式三层管道:TextStreamer → Queue → StreamingResponse
- 手写 `parse_response`:`<think>` → reasoning_content,`<tool_call>` → tool_calls
- 解释模型格式转换:`.pth` → HF Qwen3 格式 → vllm 部署
- 演示 LoRA 合并和 MoE expert stacking

## 文件清单

| 文件 | 用途 |
|---|---|
| [`ch11.ipynb`](./ch11.ipynb) | 主 notebook(教学载体,边读边跑) |
| [`serving.ipynb`](./serving.ipynb) | 精简总结(快速复习) |
| [`exercise-solutions.ipynb`](./exercise-solutions.ipynb) | 3 道习题 + 解答 |
| [`ch11.md`](./ch11.md) | 中文导读(摘要 + 文件:行引用 + 术语表) |

## 对应 minimind 源码

- `scripts/serve_openai_api.py`(252 行):FastAPI + OpenAI 兼容 API + SSE 流式 + 输出解析
- `scripts/convert_model.py`(144 行):格式转换 + LoRA 合并 + MoE expert stacking
- `scripts/eval_toolcall.py`(240 行):8 个 mock 工具 + 多轮工具调用评估

## 核心流程

```
训练侧 (.pth)                     部署侧
MiniMindForCausalLM    convert     Qwen3ForCausalLM
  state_dict        ─────────→     HF transformers
                       LoRA merge          │
                                  ┌────────┼────────┐
                                 vllm    ollama   llama.cpp
                                            │
                                    serve_openai_api.py
                                            │
                                   /v1/chat/completions
                                   SSE streaming
                                   <think> → reasoning_content
                                   <tool_call> → tool_calls
```

---

← [教程总览](../../README.md) | → [第 12 章:DPO 对齐](../ch12/01_main-chapter-code/README.md)
