# 附录 A:PyTorch 速成

> 本附录为不熟悉 PyTorch 的读者提供最小前置知识。如果你已经会用 PyTorch,可以跳过。

## 你需要掌握的 PyTorch 概念

本教程用到的 PyTorch API 非常集中。以下是按出现顺序排列的最小知识集:

### 1. Tensor(张量)

```python
import torch

# 创建
x = torch.randn(2, 3)          # 形状 (2, 3) 的正态分布随机张量
x = torch.zeros(4, 5)          # 全零
x = torch.tensor([1, 2, 3])    # 从列表

# 形状操作
x = x.view(3, 2)               # reshape(类似 numpy.reshape)
x = x.transpose(0, 1)          # 交换维度 0 和 1
x = x.unsqueeze(0)             # 增加一个维度 (2,3) → (1,2,3)

# 运算
y = x @ x.T                    # 矩阵乘法
y = torch.softmax(x, dim=-1)   # softmax
y = x.sum(dim=1)               # 沿维度 1 求和
```

### 2. nn.Module(神经网络模块)

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)  # 可学习参数

    def forward(self, x):
        return self.proj(x)  # 定义前向传播

# 使用
layer = MyLayer(768)
out = layer(x)  # 自动调用 forward
```

### 3. 训练循环

```python
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()        # 清零梯度
    loss = model(batch)          # 前向传播(返回 loss)
    loss.backward()              # 反向传播(自动求导)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()             # 更新参数
```

### 4. 设备与精度

```python
model = model.cuda()             # 移到 GPU
model = model.half()             # 半精度(float16/bfloat16)
model = model.eval()             # 推理模式(关闭 dropout)

with torch.no_grad():            # 不计算梯度(推理时)
    out = model(x)

with torch.autocast('cuda'):     # 自动混合精度
    out = model(x)
```

### 5. 保存与加载

```python
torch.save(model.state_dict(), 'model.pth')                    # 保存
model.load_state_dict(torch.load('model.pth', map_location='cpu'))  # 加载
```

## 延伸阅读

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
