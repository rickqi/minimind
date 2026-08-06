# scripts/AGENTS.md

> 本目录是 MiniMind 的"驱动层": 52 个脚本, 覆盖 WSL 训练启动、医疗数据管线、PLE 量化导出、ESP32 部署产物生成与推理服务。
> 父级约束见根 `AGENTS.md` (模型输出规范 / 多环境隔离)。本文件只记录**模板与陷阱**, 不重复命令速查。

---

## 1. 六大分类速查

| 分类 | 文件模式 | 作用 | 运行环境 |
|---|---|---|---|
| A. WSL 训练启动 | `wsl_train_*.sh` / `wsl_sft_*.sh` / `wsl_dpo_*.sh` | 调 trainer/train_*.py | WSL GPU |
| B. WSL 评估 | `wsl_eval_*.sh` (13个) | 内联 heredoc 问答评估 | WSL GPU |
| C. WSL 工具 | `wsl_export_dpo.sh` `wsl_verify_dpo.sh` `wsl_status.sh` `wsl_gpu_test.sh` | 批量导出 / 权重校验 / 看门狗 | WSL GPU |
| D. 医疗数据管线 | `build_medical_*.py` + `mix_medical.py` | 生成 dataset/*.jsonl | Windows/WSL |
| E. 导出/量化/部署 | `export_ple1.py` `quantize_ple.py` `gen_vocab_minimind.py` `rag_medical.py` `register_model.py` | PLE1/int4/vocab.h/RAG/登记 | Windows |
| F. 推理/服务/演示 | `serve_openai_api.py` `chat_api.py` `web_demo.py` `eval_toolcall.py` `convert_model.py` | API/WebUI/转换 | Windows |

## 2. WSL 训练启动模板 (A 类, 所有 wsl_train/sft/dpo 统一骨架)

```bash
cd /mnt/d/codes/minimind/trainer        # WSL 路径硬编码
export CUDA_VISIBLE_DEVICES=0           # 单卡硬编码
pkill -f "train_<X>.py"; sleep 2        # 杀旧进程
rm -f ../out/<save>.log ../out/<save>_<dim>_ple.pth ../checkpoints/<save>_<dim>_ple*.pth  # 破坏性清理
exec python3 -u train_<X>.py --use_ple 1 --ple_dim {96|128} \
    --hidden_size {256|384|512} --num_hidden_layers {6|8|8} \
    --data_path ../dataset/<set>.jsonl [--from_weight <前序权重>] \
    --save_weight <name> --save_dir ../out ... 2>&1 | tee ../out/<save>.log
```

- 输入: `dataset/*.jsonl` + `out/{前序}_{dim}_ple.pth` (续训); 输出: `out/{name}_{dim}_ple.pth` + `out/{name}.log` + `checkpoints/{name}_{dim}_ple*.pth` (断点)
- **陷阱: 脚本名 ≠ 训练器**。`wsl_train_h2_raft.sh` 实际跑的是 `train_full_sft.py` (SFT 微调)。判据以脚本内 `train_*.py` 为准, 不要信文件名。

## 3. 医疗数据管线 DAG (D 类)

```
外部源: ../esp32-ai/data_v4/corpus.txt + kb/format_data.jsonl + D:/docs/raw/{medica,临床诊疗指南全集}
  ├─[A]  build_medical_pretrain.py      -> dataset/pretrain_medical.jsonl
  ├─[B1] build_medical_sft_b1.py        -> dataset/sft_medical_b1.jsonl
  ├─[B2] build_medical_sft_b2.py (需 --api-key) -> dataset/sft_medical_b2.jsonl (可断点 out/b2_cache.json)
  ├─[pure] build_medical_sft_pure.py    -> dataset/sft_medical_pure.jsonl (B1过滤+B2)
  ├─[RAFT] build_medical_raft.py --no-evidence-ratio 0.3 --negative-ratio 0.15 --med-only -> dataset/sft_medical_raft.jsonl
  └─[mix] mix_medical.py                -> dataset/pretrain_mixed.jsonl + sft_medical_mixed.jsonl (1:2 / 1:3)
每管线产出 out/medical_*_report.json 审计报告
```

- **关键依赖: 所有医疗脚本 + RAG/RAFT 评估都读兄弟仓库 `../esp32-ai/data_v4/`, 无 esp32-ai 检出则硬失败。**

## 4. 导出/量化 I/O 约定 (E 类)

| 工具 | 输入 | 输出 | 关键细节 |
|---|---|---|---|
| `export_ple1.py` | `out/{w}_{dim}_ple.pth` | `models/{w}_h{dim}_ple1.bin` + golden | int4 group=32; GQA→MHA `repeat_interleave(2)`; golden **必须来自反量化模型** |
| `quantize_ple.py` | `out/{w}_{dim}_ple.pth` | `models/{w}_{dim}_int4_g32.pth` | **group=32 硬约束** (128 崩, 16 过拟合) |
| `gen_vocab_minimind.py` | `model/tokenizer.json` | `../esp32-ai/firmware/.../vocab.h` | 用 bytes 逆映射生成原始 UTF-8 字节 |
| `rag_medical.py` | `../esp32-ai/data_v4/kb/format_data.jsonl` | `out/rag_index.pkl` (build) | 子命令 build/query/chat |
| `register_model.py` | 文件路径参数 | 追加 `docs/MODELS.md` + `CHANGELOG.md` | 模型输出后**必跑** (见根 AGENTS.md §模型输出规范) |

## 5. 陷阱速查

1. WSL 路径硬编码 `/mnt/d/codes/minimind/...`; Python 工具双兼容 (`D:/` 与 `/mnt/d/` 回退)。
2. **训练脚本破坏性**: 启动即 `pkill` + `rm -f` 旧权重/日志/断点, 无备份。
3. 评估脚本 (B 类) 是非参数化 heredoc, 权重名/问题集写死, 不可复用; 可复用入口只有 `rag_medical.py`。
4. 输出目录 `out/ models/ checkpoints/` 全部 gitignored, 产物靠 register_model.py 登记追踪。
5. `_ple` 后缀: 所有 PLE 权重为 `{name}_{dim}_ple.pth`, 导出物对应 `_h{dim}_ple1.bin` / `_int4_g32.pth`。
