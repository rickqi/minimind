# 附件利用率 = 0% 断点分析 + 打通方案 (ATTACHMENT GAP)

> 日期: 2026-08-10 | 对象: EmailAgent 训练数据管线
> 结论: 51K 附件 markdown 完全未用于训练, 根因是 5 处数据断点; 方案 B (raw/ 加载器) 是推荐打通路径

---

## 一、数据断点技术根因 (5 处, 按数据流)

```
原始附件 (downloaded_emails/ 349K 二进制)
    │
    ├─→ markdown_converter.py ─→ raw/*.md (51K 附件) ─→ ❌ 断点#3: write-only, 零消费者
    │
    └─→ attachment_reader.py (text_content 全文, 4类型)
            │
            └─→ email_parser.py:169 ─→ ❌ 断点#1: 只存 summary(500字), 丢弃全文
                    │
                    └─→ parsed.json ─→ common.py 3加载器 ─→ ❌ 断点#2: 只读 body_text
                            │
                            └─→ 6训练管线 (A/B1/B2/B3/C/E) — 附件字段从未读取
```

| # | 断点 | 位置 | 根因 |
|---|---|---|---|
| **1** | 解析层截断 (主因) | `email_parser.py:169-174` | `attachments.append()` 只存 `summary`(500字), **丢弃 `text_content` 全文** |
| **2** | 加载层缺入口 | `common.py` 3加载器 | 全部只读 parsed.json, **无任何 raw/*.md 读取函数** |
| **3** | raw/ write-only | `markdown_converter.py:23` | **51K 附件 md 是孤儿数据**, 无反向读取者 |
| **4** | 映射可行 ✅ | `email_folder` 字段 | downloaded_emails→raw 替换即可, **但需处理文件名截断** |
| **5** | 全文可用性 | `attachment_reader.py` | text_content 仅 4 类型 (xlsx/docx/pdf/pptx); **raw/*.md 覆盖 13 类型** |

## 二、关键量化证据

| 指标 | 数值 | 含义 |
|---|---|---|
| parsed summary vs raw md | 500字 vs 1,318字 | **summary 只保留 37.9% 内容** |
| raw md 覆盖格式 | **13 种** (含 eml/msg/html/zip) | 远超 attachment_reader 4 种 |
| 断点可逆性 | email_folder→raw 路径实测存在 | **打通成本低** |

## 三、打通方案 (推荐方案 B)

### 方案 B: 新增 raw/ 加载器, 零重解析

**common.py 新增 3 函数** (L149 后):

```python
def raw_dir() -> Path:
    return path("raw")

def load_email_raw(email: Dict) -> Dict[str, str]:
    """从 raw/ 加载邮件正文+附件 markdown 全文"""
    folder = email["email_folder"].replace("downloaded_emails", "raw", 1)
    attach_mds = glob.glob(f"{folder}/attachments/*.md")  # glob 前缀匹配防截断
    return {"body": read(f"{folder}/*.txt.md"), "attachments": attach_mds}
```

**覆盖**: 13 种扩展名 (含 html/eml/msg/zip/doc, 远超 attachment_reader 4 种)
**零破坏**: parsed.json 不动, 上游解析链不改, 仅训练侧增加可选加载

### 方案 A (补充): email_parser.py:174 加 text_content 字段
覆盖未来新增数据 (需全量重解析, 代价高)

### 方案 C (备选): 重新调用 scan_attachments
无需 raw/, 但仅 4 类型 + 每次重读二进制

## 四、管线接入点

| 管线 | 位置 | 改造 |
|---|---|---|
| **C RAFT** (P0) | `stage_c_raft.py:78 _build_evidence_index` | 在 `for e in data["emails"]` 内追加附件文档到证据池 → **证据池+50%** |
| **B2 QA** (P0) | `stage_b2_qa_synthesis.py:292` | `body[:2000]` 处追加附件内容 → **~200K 新 QA 对** |
| **A pretrain** (P1) | `stage_a_pretrain.py:103-119` | chunk 循环追加附件 md → **~153K 文档** |
| **B3 tasks** (P1) | `stage_b3_tasks.py` | 附件文档做 t2 摘要/t3 行动项 |
| **E DPO** (P2) | `stage_e_dpo.py` | "引用附件回复" vs "通用回复" 偏好对 |

## 五、主要问题清单 (非断点)

| 问题 | 严重度 | 建议 |
|---|---|---|
| 附件含 PII (人名/公司) | 🔴 高 | 复用 PIIMapper 脱敏 |
| 投票/日历噪声 (5-8K) | 🟡 中 | 过滤 赞成(V) + 接受 邮件 |
| 文件名截断陷阱 | 🟡 中 | glob 前缀匹配非精确拼接 |
| B2/C 依赖 LLM (API key) | 🟡 中 | 附件 QA 需 LLM, 成本可控 |
| "other" 附件 (225/807) | 🟢 低 | raw md 已覆盖, 无需担忧 |

## 六、执行状态

- [x] 断点根因定位 (5 处)
- [x] 方案 B 设计 (common.py 3 函数)
- [x] common.py 加载器实现 (raw_dir/load_email_raw/load_attachment_markdown)
- [x] 加载器验证 (真实 email_folder → raw, 正文+表格附件加载成功)
- [x] **C RAFT 接入** (证据池 1876 篇含 543 附件, +28.9%, PII 脱敏生效)
- [x] **B2 QA 接入** (附件内容拼入 QA 源)
- [x] **A pretrain 接入** (附件 md 分块入语料)
- [ ] 全量管线重跑 (C RAFT / B2 QA 附件增强数据)
- [ ] 训练验证 (附件增强数据 → 模型)

## 七、参考

- email_parser.py: /home/EmailAgent/email_knowledge-v3/src/email_parser.py (L164-176)
- attachment_reader.py: /home/EmailAgent/email_knowledge-v3/src/attachment_reader.py (L38-45, L120-135)
- common.py: /home/EmailAgent/email_knowledge-v3/skills/training_data_gen/training_data_gen/common.py (L60-149)
- markdown_converter.py: /home/EmailAgent/email_knowledge-v3/src/markdown_converter.py (L21-38, L261-262)
- stage_c_raft.py: .../stages/stage_c_raft.py (L78-121 _build_evidence_index)
- stage_b2_qa_synthesis.py: .../stages/stage_b2_qa_synthesis.py (L262-292)
