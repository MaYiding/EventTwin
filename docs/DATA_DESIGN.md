# 数据分层存储与幂等设计（v3）

> 目标：数据源与爬取产物**按公司/时间分块**组织；清洗结果独立成层；
> 抽取、Embedding 全链路**幂等**——同内容不重复抽取、同文本不重复向量化。
> 单机实现继续 SQLite（与现有可复现链路一致），并附 PostgreSQL 生产版 DDL。

## 1. 分层模型

```
L0  RAW      data/corpus/v2/{company}.json        原始批（不可变，已入 git）
             ↑ 子 Agent 爬取产物，只增不改

L1  CLEAN    data/corpus/clean/{company}/{YYYY-MM}.json
             ↑ organize_corpus.py 产出：URL/内容去重、正文截断≤800、时间 ISO 规范
             ↑ 每条带幂等键：content_hash（正文 sha256[:16]）+ url_hash

L2  MANIFEST data/corpus/clean/manifest.json
             ↑ 每文件 sha256/条数/月份/公司 + 清洗规则版本 + 全局统计
             ↑ corpus_hash 由此派生（流水线 run manifest 的输入指纹）

L3  LEDGER   corpus_item 表（独立 SQLite：data/state/corpus_ledger.db）
             ↑ 清洗条目全局账本：content_hash UNIQUE —— 抽取幂等的全局键
             ↑ 跨公司转载：同 content_hash 复用抽取结果，仅登记来源关联

L4  EXTRACT  event_mention（intel.db，已有）
             ↑ 幂等键 = (document_version_id, run_id, local_id)

L5  EMBED    embedding_cache 表（独立 SQLite：data/state/embedding_cache.db）
             ↑ 幂等键 = (model_id, text_hash) UNIQUE
             ↑ vector 表（intel.db）已有 text_hash 列，两层共同保证不重复向量化

L6  LLM_CACHE data/state/llm_cache.db（chat + rerank）
             ↑ 幂等键 = sha256(model, messages/params)，stage 列可审计可清理
```

**为什么 clean 层独立**：原始批不可变（审计/重放），清洗规则会演进（v1→v2）；
分层后清洗规则升级只需重跑 L1（纯本地计算，零模型调用），L4/L5 缓存按 text_hash
自动部分复用。

## 2. 分块组织（L1 目录结构）

```
data/corpus/clean/
├── manifest.json
├── xiaomi/
│   ├── 2024-01.json    # 该公司该月清洗后条目（含 content_hash/url_hash/来源）
│   ├── 2024-02.json
│   └── ...
├── huawei/
│   └── ...
└── ...（30 家）
```

- 月文件内条目按 published_at 升序（确定性）；
- 无月份的条目（本批 0 条）归入 `unknown.json` 并在 manifest 标注；
- `manifest.json`：`{"clean_version": "clean-v1", "files": [{"path","sha256","company","month","n_items"}], "stats": {...}}`；
- 切换消费：`config.pipeline.corpus_dirs` 从 `v2` 换成 `data/corpus/clean`（ingest 的
  URL+内容幂等保证切换零重复；本切换在全量跑完成后执行，避免干扰在跑流水线）。

## 3. 幂等键与去重规则

| 层 | 幂等键 | 规则 |
|---|---|---|
| 爬取入库 | (source_id, norm_url) | 同 URL 同内容跳过；同 URL 新内容 → 新 document_version |
| 清洗 | content_hash（正文 sha256[:16]） | **公司内**同 content_hash 只留最早一条；**跨公司**同 content_hash 全保留但互相登记 `duplicate_of`（企业间转载，情报上要算两个主体各自的证据） |
| 抽取 | corpus_item.content_hash（全局） | 同 content_hash 的后续文档复用首个的抽取结果（`duplicate_of` 链）——同内容跨公司不重复调 LLM；这正是"转载十次只算一个独立来源组"的存储落地 |
| Embedding | (model_id, text_hash) | 文本未变不重嵌；框架文本变化 → 新 text_hash 新向量，旧向量 active=0 |
| Rerank/Chat | sha256(模型+提示+参数) | 已有 llm_cache；重放零 GPU |

清洗规则（clean-v1）：
1. `norm_url`：去 query tracking 参数、去 fragment、小写 host、去尾斜杠；
2. 正文：去首尾空白 → 截断到 800 字（句边界优先）；<100 字丢弃（登记 dropping 计数）；
3. `published_at`：统一 ISO+08:00（失败置 null 并计数）；
4. 输出字段：`url, url_hash, content_hash, title, published_at, fetched_at, source_name, language, content, duplicate_of(可空)`。

## 4. 表结构

### 4.1 SQLite（本仓实现，`intel/store/db.py` + 独立账本库）

```sql
-- 清洗条目账本（data/state/corpus_ledger.db）
CREATE TABLE corpus_item (
  content_hash TEXT NOT NULL,          -- 正文 sha256[:16]，全局抽取幂等键
  url_hash     TEXT NOT NULL,
  company      TEXT NOT NULL,          -- 分块归属
  month        TEXT,                   -- YYYY-MM 分块
  url          TEXT, title TEXT, published_at TEXT, source_name TEXT,
  content_len  INTEGER NOT NULL,
  first_seen_in TEXT NOT NULL,         -- 首次出现的 clean 文件相对路径
  duplicate_of TEXT,                   -- 同 content_hash 更早条目的 content_hash（跨公司转载链）
  extracted    INTEGER NOT NULL DEFAULT 0,   -- 抽取完成标记（流水线回写）
  document_version_id TEXT,            -- 回链 intel.db
  created_at   TEXT NOT NULL,
  PRIMARY KEY (company, content_hash)
);
CREATE INDEX ledger_by_hash ON corpus_item(content_hash);   -- 全局查重/复用
CREATE INDEX ledger_by_company ON corpus_item(company, month);

-- Embedding 缓存（data/state/embedding_cache.db）
CREATE TABLE embedding_cache (
  model_id  TEXT NOT NULL,
  text_hash TEXT NOT NULL,             -- sha256(全文)
  dim       INTEGER NOT NULL,
  vec       BLOB NOT NULL,             -- float32 L2 归一化
  created_at TEXT NOT NULL,
  PRIMARY KEY (model_id, text_hash)
);
```

### 4.2 PostgreSQL（生产版，多写者/多机部署时替换）

```sql
-- 清洗账本：全局抽取幂等（跨公司转载共享抽取结果）
CREATE TABLE corpus_item (
  id            BIGSERIAL PRIMARY KEY,
  content_hash  CHAR(16) NOT NULL,
  url_hash      CHAR(16) NOT NULL,
  company       TEXT NOT NULL,
  month         CHAR(7),
  url           TEXT NOT NULL,
  title         TEXT,
  published_at  TIMESTAMPTZ,
  source_name   TEXT,
  content_len   INTEGER NOT NULL CHECK (content_len BETWEEN 100 AND 800),
  first_seen_in TEXT NOT NULL,
  duplicate_of  CHAR(16) REFERENCES corpus_item(content_hash),
  extracted     BOOLEAN NOT NULL DEFAULT FALSE,
  document_version_id UUID,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (company, content_hash)
);
CREATE INDEX ON corpus_item (content_hash);
CREATE INDEX ON corpus_item (company, month);

-- 抽取结果按内容哈希共享（同一 content_hash 只有一份抽取产物，多处引用）
CREATE TABLE extraction_result (
  content_hash CHAR(16) PRIMARY KEY REFERENCES corpus_item(content_hash),
  mentions     JSONB NOT NULL,
  standalone   JSONB NOT NULL DEFAULT '[]',
  model_id     TEXT NOT NULL,
  prompt_hash  CHAR(16) NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  usage_count  INTEGER NOT NULL DEFAULT 1      -- 被多少文档复用（转载计数）
);

-- Embedding 缓存（生产可换 pgvector 表 + 缓存表分离）
CREATE TABLE embedding_cache (
  model_id   TEXT NOT NULL,
  text_hash  CHAR(16) NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BYTEA NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (model_id, text_hash)
);
-- 向量本体用 pgvector：
-- CREATE TABLE vec (text_hash CHAR(16) PRIMARY KEY, model_id TEXT, embedding vector(2560));

-- 分块清单（对应 manifest.json）
CREATE TABLE corpus_block (
  path     TEXT PRIMARY KEY, company TEXT, month CHAR(7),
  sha256   CHAR(64) NOT NULL, n_items INTEGER NOT NULL,
  clean_version TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now()
);
```

## 5. 流水线接入（本轮不启用，跑完当前全量后切换）

1. `ingest` 扫 `clean` 目录：按 manifest 顺序 → `corpus_item` upsert（幂等）；
2. `extract` 前：查 `extraction_result`（按 content_hash）——命中则复用（mentions 直接
   落库、`usage_count+1`），未命中才调 LLM 并写回；
3. `vectorize` 已按 text_hash 幂等，`embedding_cache` 独立库可单独清理重建；
4. 重放：`--reset` 后全链路命中三层缓存（chat/rerank + extraction_result + embedding），
   预计分钟级。

## 6. 数据源索引

见 `data/corpus/clean/SOURCES.md`：30 家 × 来源通道（东财资讯流 API / 华尔街见闻归档 /
企业官网 API / Google News RSS 解码 / 人民网搜索 / IT之家标签页等）与月份覆盖矩阵。

## v3.1 附注（2026-09-29 · 27B + Embedding-8B + Jev 三件套）

向量层随内网新服务（10.10.21.216，qwen3-embedding-8b，4096 维）回归：ANN 召回、dense 检索、
S_semantic 质心余弦全部恢复；判定职责仍全归 Jev（P_same / 灰区判别 / 检索重排），
生成职责归 qwen3.8-27b（enable_thinking=false 关思考）。reranker/Laya 不再使用。
以下为 v3.0（纯双模型、无向量）时期的历史附注：

- 召回：BM25 + 结构化通道 + 未投影补丁（`recall.py`，无 ANN）；
- S_semantic 特征：BM25 词面分池内归一（`resolve._rank_candidates`）；
- P_same 特征 / 灰区判别 / 检索重排：Jev（OpenRouter decisions，`llm.decide` / `llm.jev_choice_rank`）；
- v3.0 曾删除的 `store/vectors.py`、`store/embedding_cache.py`、`pipeline/vectorize.py`、
  `scripts/backfill_card_vectors.py` 已随 embedding-8b 回归全部恢复。
