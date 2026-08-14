# full_v2 全量训练复盘 (2026-08-13 ~ 14)

> 首次在 GB10 (Grace Blackwell 统一内存) 上完成 EmailAgent full_v2 (169万行) 全量训练,
> 覆盖 minimind 训练技能的两种手段 (Dense / PLE) × 两种架构 (H1 / H2), 含 PLE 部署链。
> 配套 `docs/MODELS.md` 第九章登记 + `CHANGELOG.md` [Unreleased]。

## 一、训练矩阵与结果

| 架构 | 手段 | 预训练 loss | SFT loss | 部署产物 | 用时 |
|---|---|---|---|---|---|
| H1 (d256/l6) | Dense | 0.645 | 0.031 | — | ~3h |
| H1 (d256/l6) | **PLE (ple96)** | 0.681 | **0.0225** | **PLE1 6.31MB + int4 (deg+0.116) + golden** | ~3h |
| H2 (d384/l8) | Dense | 0.591 | 0.017 | — | ~9h |
| H2 (d384/l8) | **PLE (ple128)** | **0.552** | **0.0195** | **PLE1 14.73MB + int4 (deg+0.027) + golden** | ~8h |

**矩阵已补完**: 两手段 × 两架构 4 组合全部完成 full_v2 (169万行) 全量训练。
DPO (H1 Dense): loss 0.54 → **0.0061** (max_seq_len 1024, 陷阱#11 预检通过)。

**注意**: H2 PLE1 14.73MB 逼近 ESP32 model 分区 14.5MB 上限, 烧录前需核对分区表, 超限仅 PC/树莓派部署。

## 二、关键结论

### 1. PLE ≈ Dense (训练期等价, 部署期 PLE 胜出)
- H1 PLE 预训练 loss 0.681 vs Dense 0.645 (PLE 略高, 容量分到查找表); SFT PLE 0.0225 略优于 Dense 0.031
- **核心价值在部署**: PLE 提供 int4 量化 (deg +0.116) + PLE1 扁平格式 (6.31MB, flash 驻留), Dense 无此链路
- 印证 SKILL.md "PLE 训练期等价, 部署期提供量化+PLE1"

### 2. 大模型容量优势 (H2 > H1)
- H2 预训练 0.591 < H1 0.645, SFT 0.017 < 0.031, logits_std 更高 → 大模型收敛更好
- 符合"模型越大、数据越大、所需 epoch 越少"

### 3. 3-epoch 决策 (vs minimind-3 的 1 epoch)
- H1 仅 6.66M (minimind-3 主线 64M 的 1/10), 小模型需多 epoch 补偿容量
- 实测 loss 跨 epoch 持续下降 (1ep 0.92 → 3ep 0.645), 1 epoch 欠拟合, 3 epoch 合理
- 依据: 模型容量比例 + SKILL.md EmailAgent 历史既定 3 epoch + 实测未饱和

### 4. DPO 对小模型短生成无效
- SFT vs DPO 问答输出几乎一致 (loss 0.006 但行为不变)
- 印证"偏好对齐 ≠ 生成内容差异", DPO 需更大数据/更长生成才显优势

## 三、工程方法学 (本次新增)

### 硬件自动评估 (`hardware_profile.py`)
- 探测 GPU/统一内存/CPU 核数 → 微基准扫描 batch_size/num_workers 拐点
- 发现 GB10 统一内存 112GB (显存非瓶颈) + torch.compile 提速 40%
- 找到"tiny 模型 kernel 启动开销是瓶颈, 非 GPU 算力非数据加载"

### 自主训练链 (`chain_*.sh`, 6 个)
- setsid 脱离会话, 无人值守串行: pretrain → verify → SFT → verify → eval (→ deploy)
- 解决"长训练被 bash 工具超时连带 SIGTERM 杀掉"问题 (setsid 新会话)

### 陷阱#11 前置预检 (DPO mask 截断)
- DPO 前抽样 `DPODataset[i]['mask_chosen'].sum()>0`, 防 loss 恒 ln2 静默空转
- max_seq_len=1024 时 5/5 安全, 512 有截断 → 选 1024

## 四、踩过的坑

| 坑 | 现象 | 修复 |
|---|---|---|
| nohup 被 bash 工具超时杀 | SIGTERM 连带杀后台进程 | 改用 setsid 脱离新会话 |
| 部署链等"文件存在"误触 | SFT save_interval 提前产中间权重, 部署跑在中间件 + 并发抢 GPU OOM | 改为等"进程退出"; 或一体化串行脚本 |
| prepare_email_data 硬编码 mini | 忽略 full_v2 (169万), 只用 mini (48K) | 改为优先 full_v2 回退 mini |
| llama-server 占 22GB GPU | 训练 CUDA OOM | 停掉 llama-server 释放显存 |
| 桶名笔误 | ins-kq6zzwo (少7) → NoSuchBucket | 实际为 ins-kq6zz7wo |

## 五、性能基准 (GB10, bfloat16, torch.compile)

| 配置 | samp/s | 单 epoch (169万行) |
|---|---|---|
| H1 Dense (d256/l6, batch64) | ~496 | ~57 min |
| H1 PLE (d256/l6/ple96, batch32) | 385 | ~73 min |
| H2 Dense (d384/l8, batch32) | 180 | ~156 min |
| H2 PLE (d384/l8/ple128, batch32) | 231 | ~122 min |

注: GB10 统一内存架构, nvidia-smi 显存查询返回 N/A (正常); torch.compile mode='reduce-overhead' 对 tiny 模型提速 40%。

## 六、产物清单

### 训练权重 (out/, gitignored)
- `email_pretrain_3ep_256.pth` / `email_sft_dense_h256_256.pth` / `email_dpo_h256_256.pth` (H1 Dense 全链)
- `email_pretrain_h2_384.pth` / `email_sft_h2_384.pth` (H2 Dense)
- `email_pretrain_h1ple_256_ple.pth` / `email_sft_h1ple_256_ple.pth` (H1 PLE)
- `email_pretrain_h2ple_384_ple.pth` / `email_sft_h2ple_384_ple.pth` (H2 PLE)

### 部署产物 (models/, gitignored)
- `email_sft_h1ple_h256_ple1.bin` (6.31MB) + golden + int4 (H1 PLE 可部署件)
- `email_sft_h2ple_h384_ple1.bin` (14.73MB) + golden + int4 (H2 PLE 可部署件, ⚠️逼近 ESP32 分区上限)

### 交付包
- `/tmp/h1ple_esp32_delivery.tar.gz` (12MB) — H1 PLE 供 esp32-ai convert + verify

## 七、后续待办
- [x] H2 PLE 全管线完成 (2026-08-14 18:36, pretrain 0.552 / SFT 0.0195 / PLE1 14.73MB) — 已登记 + COS 备份 + 交付包
- [ ] esp32-ai 侧: convert_h2 + verify_h2 (PASS 阈值 maxabs<0.02) — 非本项目职责; H2 PLE1 14.73MB 需先核对分区表 (14.5MB 红线)
- [ ] DPO 在 H2 / 更长生成的效果验证 (H1 短生成无效)
- [ ] on-policy 硬负样本 (替代模板负样本, 解决长度奖励坍缩)

## 八、COS 备份索引 (backups/email-pretrain/)

| 备份 | 内容 | 大小 |
|---|---|---|
| `email_h1ple_deploy_20260814_133332.zip` | H1 PLE 部署件 (models/) | 11.0MB |
| `email_full_v2_weights_20260814_133338.zip` | 训练权重快照 (H2 PLE 未完时) | 213.2MB |
| `email_ple_deploy_full_20260814_192231.zip` | 全部部署件 (H1+H2 PLE) | 36.5MB |
| `email_full_v2_weights_final_20260814_192258.zip` | **全部最终权重** (含 H2 PLE 完成) | 260.0MB |

交付包: `/tmp/h1ple_esp32_delivery.tar.gz` (12MB) + `/tmp/h2ple_esp32_delivery.tar.gz` (26MB)
