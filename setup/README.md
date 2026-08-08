# 环境准备

> 本教程需要 minimind 源码 + Python 环境 + (可选)GPU + (可选)预训练权重。

## 1. 克隆 minimind

```bash
git clone https://github.com/rickqi/minimind.git
cd minimind
git checkout master   # 教程引用 master 分支的代码
```

> 教程中的 `文件:行` 引用基于 master 分支 commit `67f114a` (混合引用:函数名为主锚,`~行号`+`@67f114a` 为辅)。如果你的版本不同,行号可能略有偏移,用函数名定位即可。
>
> 本仓库是 `jingyaogong/minimind` 的 fork。教程教默认模式 (`use_ple=False`),行为与上游原版一致。

## 2. Python 环境

推荐 Python 3.10+,建议用 conda 或 venv 隔离:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

安装 minimind 依赖(`torch` 需按你的 CUDA 版本单独装):

```bash
# 先装 PyTorch(按 https://pytorch.org/get-started/locally/ 选你的版本)
pip install torch  # CPU 版;GPU 版见官方指引

# 再装其余依赖
pip install -r requirements.txt
```

核心依赖(已在 `requirements.txt` 中):

| 包 | 版本 | 用途 |
|---|---|---|
| `transformers` | 4.57.6 | tokenizer / 模型基类 |
| `trl` | 0.13.0 | (参考对比用,minimind 核心算法不依赖) |
| `datasets` | 3.6.0 | 数据加载 |
| `tiktoken` | 0.10.0 | (对比用) |
| `streamlit` | 1.50.0 | web demo |
| `wandb` / `swanlab` | — | 训练可视化 |

## 3. GPU(可选但强烈推荐)

- **CPU 可跑**:ch1-7(模型部分)和推理可以纯 CPU 跑(慢)。
- **训练需 GPU**:ch8+ 的训练阶段建议至少 1 张 8GB+ 显存的 GPU(如 RTX 3060 / 3090)。
- minimind 设计目标:**单张 3090,2 小时跑完 1 epoch SFT,成本约 ¥3**。

## 4. 下载预训练权重(可选)

如果想直接体验推理(ch1)而不用自己训练,从 HuggingFace / ModelScope 下载 minimind 已发布的权重:

- HuggingFace: https://huggingface.co/jingyaogong/minimind-3
- ModelScope: https://modelscope.cn/models/gongjy/minimind-3

下载后放到 `minimind/out/` 目录。

## 5. 下载数据(训练阶段需要)

按 `minimind/dataset/dataset.md` 的指引,从 ModelScope/HF 下载到 `minimind/dataset/`。

**最小可跑组合**(README 称为"Zero 组合"):
- `pretrain_t2t_mini.jsonl`(1.2 GB)
- `sft_t2t_mini.jsonl`(1.6 GB)

## 6. 验证安装

```bash
cd minimind
python eval_llm.py --weight full_sft --load_from model
# 选择 [0] 自动测试,看模型是否输出回复
```

如果看到模型流式输出中文回复,环境就绪。

---

## 常见问题

- **`ModuleNotFoundError: model.model_minimind`**:确保在 `minimind/` 根目录运行,或设置 `PYTHONPATH`。
- **CUDA OOM**:减小 batch_size 或用 `--device cpu`。
- ** tokenizer 加载失败**:确认 `model/tokenizer.json` 存在(随仓库一起)。
