# minimind 代码分析报告

> 本报告是教程的"地图":把 minimind 代码库的结构、架构、训练流水线一次性梳理清楚。
> 所有 `文件:行` 引用以 minimind commit `512eed0` 为准。
> 教程每一章都会回指这里的行号。

---

## 1. 目录地图

minimind 根目录 `/home/minimind`,核心 Python 文件共 22 个:

| 顶层 | 用途 |
|---|---|
| `model/` | 纯 PyTorch Transformer 实现 + LoRA 模块 + 内置 tokenizer |
| `trainer/` | 全部训练脚本(10 个 `.py`):pretrain / SFT / LoRA / DPO / PPO / GRPO / 蒸馏 / Agent-RL + rollout engine + 共享 utils |
| `scripts/` | 推理 / 服务 / 评估工具(5 个 `.py`):OpenAI 兼容 API / Streamlit web demo / 模型转换 / 工具调用评估 / SDK 客户端 |
| `dataset/` | 数据集类(5 种);`.jsonl` 数据文件按 `dataset.md` 下载到此 |
| `images/` | 架构图(`LLM-structure.jpg` / `LLM-structure-moe.jpg` / loss 曲线 / 雷达图) |
| `eval_llm.py` | 标准 CLI 推理入口(94 行) |
| `README.md` / `README_en.md` | 中 / 英项目文档(≈2000 行,含所有 RL 算法的数学推导) |

**完整 .py 清单(22 个)**

```
model/model_minimind.py          # 整个模型:config + transformer + generate(288 行)
model/model_lora.py              # LoRA 层 + apply/load/save/merge
model/__init__.py                # 空
dataset/lm_dataset.py            # PretrainDataset/SFTDataset/DPODataset/RLAIFDataset/AgentRLDataset
dataset/__init__.py              # 空
trainer/trainer_utils.py         # lr schedule / DDP / checkpoint / reward model wrapper
trainer/train_pretrain.py        # 预训练
trainer/train_full_sft.py        # 全参 SFT
trainer/train_lora.py            # LoRA SFT
trainer/train_dpo.py             # DPO
trainer/train_ppo.py             # PPO(Actor+Critic+Ref+Reward)
trainer/train_grpo.py            # GRPO / CISPO
trainer/train_distillation.py    # 白盒知识蒸馏
trainer/train_agent.py           # 智能体 RL(多轮工具)
trainer/train_tokenizer.py       # BPE tokenizer 训练(参考用)
trainer/rollout_engine.py        # 可插拔 rollout 抽象(Torch vs SGLang)
scripts/serve_openai_api.py      # FastAPI OpenAI 兼容服务
scripts/web_demo.py              # Streamlit 聊天 UI
scripts/convert_model.py         # torch↔transformers 转换 + LoRA merge
scripts/eval_toolcall.py         # 工具调用评估
scripts/chat_api.py              # OpenAI SDK 客户端 demo
```

---

## 2. 模型架构(`model/model_minimind.py`,288 行)

整个模型在**一个 288 行的文件**里 —— 极适合教学。

### 2a. `MiniMindConfig`(第 10–45 行)

继承 `transformers.PretrainedConfig`,`model_type = "minimind"`。默认值:

| 字段 | 默认 | 行 | 说明 |
|---|---|---|---|
| `hidden_size` | 768 | 12 | d_model |
| `num_hidden_layers` | 8 | 12 | 深度 |
| `vocab_size` | 6400 | 18 | 极小(教学设计) |
| `num_attention_heads` | 8 | 22 | q 头数 |
| `num_key_value_heads` | 4 | 23 | **GQA**(2:1) |
| `head_dim` | 768//8 = 96 | 24 | |
| `intermediate_size` | ⌈768·π/64⌉·64 = 2432 | 26 | SwiGLU FFN 宽度 |
| `max_position_embeddings` | 32768 | 27 | 训练时短;YaRN 外推 |
| `rope_theta` | 1e6 | 29 | 高 base,适配短训练 |
| `tie_word_embeddings` | **True** | 30 | 输入/输出词表共享 |
| `flash_attn` | True | 21 | 用 `scaled_dot_product_attention` |
| `use_moe` | False | 12 | 切换 MoE FFN |
| MoE: `num_experts` | 4 | 41 | |
| MoE: `num_experts_per_tok` | 1 | 42 | top-1 路由 |
| MoE: `router_aux_loss_coef` | 5e-4 | 45 | 负载均衡 |

> **结构对齐 Qwen3 / Qwen3-MoE** —— 在 `scripts/convert_model.py:57-71` 可见权重直接映射进 `Qwen3Config` / `Qwen3MoeConfig`。

### 2b. 构件

| 构件 | 行 | 要点 |
|---|---|---|
| **RMSNorm** | 50–60 | `x · rsqrt(mean(x²)+eps)`,上转 float32 再转回 |
| **RoPE `precompute_freqs_cis`** | 62–78 | 按 `head_dim` 预算 cos/sin;`torch.cat([cos,cos])` 配合 `rotate_half` |
| **`apply_rotary_pos_emb`** | 80–84 | rotate-half |
| **`repeat_kv`** | 86–89 | GQA 扩展:`n_rep = q_heads // kv_heads = 2` |
| **Attention** | 91–134 | q/k/v/o proj 无 bias;**QK-Norm**(`q_norm`/`k_norm` RMSNorm on head_dim);flash 路径 + 手写路径;**KV cache** 支持 |
| **FeedForward** | 136–146 | **SwiGLU**:`down_proj(silu(gate_proj(x)) · up_proj(x))` |
| **MOEFeedForward** | 148–176 | `gate` 线性路由;top-k;`index_add_` 派发;负载均衡 aux_loss |
| **MiniMindBlock** | 178–194 | pre-norm 残差:`h + attn(ln(h))` / `h + mlp(ln(h))` |

### 2c. 模型与 forward

- **`MiniMindModel`**(196–232):`embed_tokens` + N 个 block + final `norm`;RoPE buffer 非 persistent;forward 从 KV cache 算 `start_pos` 切正确 RoPE 窗口;累加 MoE aux_loss。
- **`MiniMindForCausalLM`**(234–288):继承 `PreTrainedModel + GenerationMixin`;`_tied_weights_keys` 告知 HF 绑定关系;`lm_head` 与 `embed_tokens` 权重共享;`forward` 支持 `logits_to_keep`(RL 分块解码);CE loss `ignore_index=-100`。
- **`generate`**(256–288):**手写采样器**(覆盖 `GenerationMixin.generate`)—— KV cache + temperature + repetition_penalty + top_k + top_p + multinomial + EOS mask + streamer。

**一次 forward 的张量流:**
```
input_ids [B,T]
 → embed_tokens → [B,T,H]
 → +dropout
 → for each block: { RMSNorm→Attention(+残差)→RMSNorm→FFN(+残差) }
 → final RMSNorm → [B,T,H]
 → lm_head → logits [B,T,V]
 → CE(shift by 1) → loss
```

---

## 3. Tokenizer 与数据

### 3a. Tokenizer
- **类型**:HuggingFace `PreTrainedTokenizerFast` 包 **BPE + ByteLevel**(不是 sentencepiece / tiktoken)。
- **文件**:`model/tokenizer.json` + `model/tokenizer_config.json`。
- **词表 = 6400**,其中 36 个特殊/保留 token(`<|endoftext|>`=0 pad/unk,`<|im_start|>`=1 bos,`<|im_end|>`=2 eos,`<tool_call>`=21,`<think>`=25 等)。
- **chat_template**:Jinja2,渲染为 `<|im_start|>role\n...content...<|im_end|>\n`,可注入 tool schema 与 `<think>` 标签。

### 3b. 数据集类(`dataset/lm_dataset.py`)

| 类 | 行 | 格式 | loss masking |
|---|---|---|---|
| `PretrainDataset` | 37–55 | `{"text":"..."}` → `[bos]+tokens+[eos]` | pad → -100 |
| `SFTDataset` | 58–119 | `{"conversations":[...]}` → `apply_chat_template` | **只 assistant 段不 mask,prompt → -100**(`generate_labels` 88–104) |
| `DPODataset` | 122–192 | `{"chosen":[...],"rejected":[...]}` | 只 assistant 段标 1 |
| `RLAIFDataset` | 195–224 | SFT 格式但 assistant 内容留空 | (RL 自算) |
| `AgentRLDataset` | 226–252 | 加 `gt` 字段 | 原始 messages+tools+gt |

---

## 4. 训练流水线(8 个阶段,共用一套骨架)

**共享骨架**(`trainer_utils.py`):`init_distributed → config → autocast → wandb/swanlab → init_model → dataset → AdamW + get_lr 余弦 → resume → compile/DDP → epoch 循环(SkipBatchSampler)`。

| 阶段 | 脚本 | 数据 | loss | lr | from_weight |
|---|---|---|---|---|---|
| **A 预训练** | `train_pretrain.py` | `pretrain_t2t_mini.jsonl` | CE(+aux) | 5e-4 | none(从零) |
| **B 全参 SFT** | `train_full_sft.py` | `sft_t2t_mini.jsonl` | CE(answer-only mask) | 1e-5 | pretrain |
| **C LoRA** | `train_lora.py` | `lora_medical.jsonl` | CE(只训 lora 参数) | 1e-4 | full_sft |
| **D DPO** | `train_dpo.py` | `dpo.jsonl` | `-logsigmoid(β·Δ)` | 4e-8 | full_sft |
| **E 蒸馏** | `train_distillation.py` | sft 数据 | `α·CE+(1-α)·T²·KL` | 5e-6 | (teacher→student) |
| **F PPO** | `train_ppo.py` | `rlaif.jsonl` | clipped policy+value+KL | 3e-7 | full_sft |
| **G GRPO/CISPO** | `train_grpo.py` | `rlaif.jsonl` | CISPO / GRPO switch | 3e-7 | full_sft |
| **H Agent-RL** | `train_agent.py` | `agent_rl.jsonl` | CISPO/GRPO(多轮) | 3e-7 | full_sft |

**关键设计点**(教程会重点讲):
- **LoRA 只挂 `o_proj`**:`model_lora.py:23` 的 `in_features == out_features` 过滤跳过了非方形的 q/k/v proj —— 刻意简化。
- **SFT answer-only masking**:`lm_dataset.py:88-104` 的 `generate_labels` 是让模型学"assistant"而非"user"的关键机制。
- **统一 PO 框架**(README 910–939 行):DPO/PPO/GRPO/CISPO = 同一目标的三项可替换(`policy_term·advantage − KL`)。

---

## 5. 推理 / 评估

- **`eval_llm.py`**(94 行):CLI 推理入口。`init_model` 两种加载路径(torch `.pth` vs HF `from_pretrained`);`apply_chat_template(open_thinking=...)`;`generate` 用 KV cache + `TextStreamer` 流式;默认 `temperature=0.85, top_p=0.95`。
- **`serve_openai_api.py`**:FastAPI on 8998;`/v1/chat/completions`;流式 SSE;解析 `<think>` → `reasoning_content`、`<tool_call>` → `tool_calls`(OpenAI 兼容)。
- **`convert_model.py`**:`convert_torch2transformers` 把 minimind 权重映射进 `Qwen3ForCausalLM`(llama.cpp / vllm / ollama 可用);`convert_merge_base_lora` 合并 LoRA。

---

## 6. 学习顺序(依赖序)

```
Config → Tokenizer → 构件(RMSNorm/RoPE/Attn/FFN) → Block+backbone+forward → generate
  → Pretrain loop → SFT loop → LoRA → Inference
  → DPO → Distillation → rollout_engine → PPO → GRPO → Agent
  → convert/serve/web_demo
```

---

## 7. 教程章节映射

| 章 | minimind 主文件 | 核心行号区间 |
|---|---|---|
| 1 | `eval_llm.py` | 全文 1–94 |
| 2 | `train_tokenizer.py` + `tokenizer_config.json` | — |
| 3 | `model_minimind.py` | 10–45 |
| 4 | `model_minimind.py` | 50–134 |
| 5 | `model_minimind.py` | 136–194 |
| 6 | `model_minimind.py` | 196–253 |
| 7 | `model_minimind.py` | 256–288 |
| 8 | `lm_dataset.py:37-55` + `train_pretrain.py` + `trainer_utils.py` | — |
| 9 | `lm_dataset.py:58-119` + `train_full_sft.py` | — |
| 10 | `model_lora.py` + `train_lora.py` | — |
| 11 | `serve_openai_api.py` + `convert_model.py` | — |
| 12 | `lm_dataset.py:122-192` + `train_dpo.py` | — |
| 13 | `rollout_engine.py` + `train_ppo.py` + `train_grpo.py` | — |
| 14 | `train_agent.py` + `lm_dataset.py:226-252` | — |
| 15 | `train_distillation.py` + `model_minimind.py:148-176` | — |
