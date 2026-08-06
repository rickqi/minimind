# trainer/AGENTS.md

> 本目录是 MiniMind 的"训练引擎": 11 个文件, 覆盖 Pretrain / SFT / DPO / RLAIF(GRPO/PPO/CISPO) / Agentic RL / LoRA / 蒸馏 / Tokenizer 全链路。
> 父级约束见根 `AGENTS.md` (模型输出规范 / 多环境隔离 / 精确命令速查)。本文件只记录训练侧**公共契约与陷阱**。

---

## 1. 文件清单

| 文件 | 用途 | 默认 from_weight / save_weight |
|---|---|---|
| `trainer_utils.py` | 共享工具: init_model / lm_checkpoint / get_lr / init_distributed_mode / SkipBatchSampler / LMForRewardModel | — |
| `train_pretrain.py` | 预训练(从零) | none → `pretrain` |
| `train_full_sft.py` | 全参 SFT(RAFT/医疗 SFT 靠 CLI 覆盖参数复用此脚本) | `pretrain` → `full_sft` |
| `train_dpo.py` | DPO 偏好优化(双模型 policy+ref, 额外 `--beta`) | `full_sft` → `dpo` |
| `train_grpo.py` | GRPO/CISPO (`--loss_type cispo`) | `full_sft` → `grpo` |
| `train_ppo.py` | PPO(Actor+Critic) | `full_sft` → `ppo_actor` |
| `train_agent.py` | Agentic RL(多轮工具调用) | `full_sft` → `agent` |
| `train_lora.py` | 手写 LoRA(无 peft 依赖, 合并用 scripts/convert_model.py) | `full_sft` → `lora_xxx` |
| `train_distillation.py` | 白盒蒸馏(teacher+student, CE+KL) | `full_sft` → 按 student |
| `train_tokenizer.py` | BPE 分词器训练(仅参考, README 不建议重训) | — |
| `rollout_engine.py` | RL 训推分离抽象(Torch / SGLang 后端) | — |

## 2. 公共 CLI 契约(每个 train_*.py 独立重复声明 argparse, 无共享基类)

- 架构: `--hidden_size` `--num_hidden_layers` `--use_moe` `--use_ple` `--ple_dim`
- I/O: `--save_dir`(默认 ../out) `--save_weight` `--data_path` `--from_weight`(或 'none') `--from_resume`
- 优化: `--epochs` `--batch_size` `--learning_rate` `--accumulation_steps` `--grad_clip` `--max_seq_len`
- 环境: `--device` `--dtype`(bfloat16) `--use_compile` `--use_wandb`(swanlab 别名) `--num_workers`
- DDP: 无标志, 靠 `RANK` 环境变量检测, `torchrun --nproc_per_node N train_xxx.py`

## 3. 必守模板(新增/修改训练脚本时逐字复制)

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL 冲突规避
```

- `use_ple` **不是** MiniMindConfig 构造参数: 必须构造后设属性 `lm_config.use_ple = True` + `lm_config.ple_dim = N`
- 主流程 9 步: init_distributed → setup_seed → makedirs+lm_config(+resume 探测) → autocast → wandb → init_model+dataset+optimizer → restore → compile(先)**再** DDP(后) → epoch 循环 + SkipBatchSampler → destroy_process_group
- 混合精度: `GradScaler` 只在 `--dtype float16` 启用; bfloat16 用 nullcontext, 勿加 scaler

## 4. 输出双目录约定

| 目录 | 内容 | 命名 |
|---|---|---|
| `../out/` | 最终权重(半精度 state_dict, 覆盖写) | `{save_weight}_{hidden_size}{suffix}.pth` |
| `../checkpoints/` | 断点包(模型+优化器+epoch+step+wandb_id) | `{save_weight}_{hidden_size}{suffix}_resume.pth` |

- 后缀优先级 `_ple` > `_moe` > (无); `init_model` 加载用 `strict=False`(PLE/MoE 部分加载容忍)
- `--from_weight` 加载 out/ 半精度权重; `--from_resume` 加载 checkpoints/ 完整训练态(跨 GPU 自动按 world_size 缩放 step)
- 保存前须 `getattr(raw_model, '_orig_mod', raw_model)` 解包 torch.compile

## 5. ⚠️ 已知缺陷: 5 个脚本缺 PLE 后缀

- ✅ PLE 感知: `train_pretrain.py` `train_full_sft.py` `train_dpo.py`
- ❌ PLE 盲(仅 `_moe`): `train_grpo.py` `train_ppo.py` `train_agent.py` `train_lora.py` `train_distillation.py`

**后果**: 对 PLE 模型跑 GRPO/PPO/Agent/LoRA/蒸馏, 会存出 `out/grpo_768.pth`(无 `_ple`), 而 `init_model` 找 `grpo_768_ple.pth` → `--from_weight` 续训 FileNotFoundError; `--from_resume` 不受影响(lm_checkpoint 用 `_model_suffix` 正确)。当前医疗管线只用 pretrain+SFT+RAFT, 不受影响; **PLE+RL 暂不支持**, 除非修复这 5 处内联后缀。

## 6. RL 专属约定

- rollout 抽象: `create_rollout_engine('torch'|'sglang', ...)`; 策略权重经 `update_policy()` 同步
- SGLang 需先另起 server, 且要 transformers 格式模型目录(非 out/*.pth, 必须先转换)
- RL 脚本用真 `CosineAnnealingLR`(经 `scheduler=` 传入 lm_checkpoint), 与 SFT 的手动 `get_lr`(余弦 0.1~1.0 区间)不同
