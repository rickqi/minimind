# 附录 D:训练循环增强

> minimind 的训练循环包含一些「锦上添花」的工程技巧。本附录补充说明。

## 1. 梯度裁剪(Gradient Clipping)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

防止梯度爆炸:如果梯度的 L2 范数超过 `max_norm`,按比例缩放回去。minimind 所有训练脚本都用 `max_norm=1.0`。

## 2. 余弦学习率调度

minimind 的 `get_lr`(trainer_utils.py:40)实现了带 warmup 的余弦衰减:

$$\text{lr}(t) = \text{lr}_{\text{base}} \cdot (0.1 + 0.45 \cdot (1 + \cos(\pi \cdot t / T)))$$

- 前 10%:lr 在底部(10% × base_lr)
- 中间:余弦上升再下降
- 末尾:回到底部

## 3. 混合精度训练(AMP)

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

bfloat16 比 float16 更稳定(指数位更多),现代 GPU(3090+)原生支持。

## 4. 分布式训练(DDP)

```python
torch.distributed.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

minimind 支持单机多卡训练。DDP 自动处理梯度同步。

## 5. 断点续训

minimind 的 `lm_checkpoint` 保存了完整的训练状态:
- model state_dict
- optimizer state_dict
- scaler state(如果是 AMP)
- 当前 epoch 和 step
- wandb run id

恢复时 `SkipBatchSampler` 会跳过已训练的 batch,确保数据顺序一致。

## 6. torch.compile

```python
model = torch.compile(model)
```

PyTorch 2.0+ 的 JIT 编译,可加速 10-30%。注意:LoRA 的 monkey-patch 与 compile 不兼容(train_lora.py 会禁用它)。
