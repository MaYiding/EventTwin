# -*- coding: utf-8 -*-
"""SQLite 存储层：schema 定义、连接管理、事务助手。

表清单对齐架构文档 04 §15.1（PostgreSQL → SQLite 单机适配，同表名同职责）：
source / raw_snapshot / document / document_version / source_lineage /
entity / entity_alias / event_mention / event_cluster / cluster_membership /
cluster_version / process / assertion / assertion_evidence /
slot_selection_history / semantic_relation / resolution_decision /
pipeline_run / outbox / projection_state / queue(消息总线) /
pipeline_event(可观测) / change_record / subscription / delivery / vector。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 来源与原文账本 ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS source (
  source_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  adapter TEXT NOT NULL DEFAULT 'corpus',
  homepage TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL UNIQUE,
  uri TEXT,
  fetched_at TEXT,
  bytes INTEGER NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS document (
  document_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES source(source_id),
  origin_id TEXT NOT NULL,
  canonical_url TEXT,
  access_scope TEXT NOT NULL DEFAULT 'public',
  created_at TEXT NOT NULL,
  UNIQUE (source_id, origin_id)
);

CREATE TABLE IF NOT EXISTS document_version (
  document_version_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES document(document_id),
  version_no INTEGER NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES raw_snapshot(snapshot_id),
  title TEXT,
  published_at TEXT,
  fetched_at TEXT,
  parse_version TEXT,
  normalized_text TEXT,
  status TEXT NOT NULL DEFAULT 'captured'
    CHECK (status IN ('captured','parsed','extracted','resolved','indexed','published',
                     'duplicate_of','failed')),
  dedupe_of TEXT,
  note TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (document_id, version_no)
);
CREATE INDEX IF NOT EXISTS dv_status ON document_version(status);
CREATE INDEX IF NOT EXISTS dv_hash ON document_version(snapshot_id);

CREATE TABLE IF NOT EXISTS source_lineage (
  document_version_id TEXT PRIMARY KEY REFERENCES document_version(document_version_id),
  lineage_group TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'original' CHECK (kind IN ('original','reprint','unknown')),
  based_on_hash TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lineage_group_idx ON source_lineage(lineage_group);

-- 实体 ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity (
  entity_id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN ('company','brand','product','person','other')),
  canonical_name TEXT NOT NULL,
  norm_key TEXT NOT NULL UNIQUE,
  note TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT,
  deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS entity_alias (
  alias_id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entity(entity_id),
  alias TEXT NOT NULL,
  norm_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (entity_id, norm_key)
);
CREATE INDEX IF NOT EXISTS alias_norm ON entity_alias(norm_key);

-- 事件提及 / 事件簇 / 成员 / 版本 --------------------------------------------
CREATE TABLE IF NOT EXISTS event_mention (
  mention_id TEXT PRIMARY KEY,
  document_version_id TEXT NOT NULL REFERENCES document_version(document_version_id),
  run_id TEXT,
  local_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  event_type_raw TEXT,
  event_phase TEXT,
  action TEXT,
  actor_json TEXT NOT NULL DEFAULT '[]',
  object_json TEXT NOT NULL DEFAULT '[]',
  event_time_lower TEXT,
  event_time_upper TEXT,
  time_precision TEXT,
  time_basis TEXT,
  scope_json TEXT NOT NULL DEFAULT '{}',
  claims_json TEXT NOT NULL DEFAULT '[]',
  evidence_json TEXT NOT NULL DEFAULT '[]',
  missing_json TEXT NOT NULL DEFAULT '[]',
  correction_hint INTEGER NOT NULL DEFAULT 0,
  frame_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'valid'
    CHECK (status IN ('valid','invalid','superseded')),
  invalid_reason TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (document_version_id, run_id, local_id)
);
CREATE INDEX IF NOT EXISTS mention_status ON event_mention(status);
CREATE INDEX IF NOT EXISTS mention_type ON event_mention(event_type);

CREATE TABLE IF NOT EXISTS event_cluster (
  cluster_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL CHECK (version > 0),
  event_type TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'provisional'
    CHECK (state IN ('provisional','resolved','ambiguous','redirected','split','deleted')),
  process_id TEXT,
  frame_json TEXT NOT NULL DEFAULT '{}',
  event_time_lower TEXT,
  event_time_upper TEXT,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  card_text TEXT NOT NULL DEFAULT '',
  card_hash TEXT,
  centroid BLOB,
  redirect_to TEXT,
  deleted_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cluster_type_time ON event_cluster(event_type, event_time_lower);
CREATE INDEX IF NOT EXISTS cluster_state ON event_cluster(state);

CREATE TABLE IF NOT EXISTS cluster_membership (
  membership_id TEXT PRIMARY KEY,
  mention_id TEXT NOT NULL REFERENCES event_mention(mention_id),
  cluster_id TEXT NOT NULL REFERENCES event_cluster(cluster_id),
  decision_id TEXT NOT NULL,
  added_at TEXT NOT NULL,
  removed_at TEXT,
  removed_reason TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_cluster_per_mention
  ON cluster_membership(mention_id) WHERE removed_at IS NULL;
CREATE INDEX IF NOT EXISTS membership_by_cluster
  ON cluster_membership(cluster_id, added_at);

CREATE TABLE IF NOT EXISTS cluster_version (
  cluster_id TEXT NOT NULL REFERENCES event_cluster(cluster_id),
  version INTEGER NOT NULL,
  decision_id TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  PRIMARY KEY (cluster_id, version)
);

CREATE TABLE IF NOT EXISTS process (
  process_id TEXT PRIMARY KEY,
  family TEXT NOT NULL,
  object_entity_id TEXT REFERENCES entity(entity_id),
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (family, object_entity_id)
);

-- 断言与双时间选择 -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS assertion (
  assertion_id TEXT PRIMARY KEY,
  slot_key TEXT NOT NULL,
  subject_entity_id TEXT NOT NULL REFERENCES entity(entity_id),
  predicate TEXT NOT NULL,
  scope_json TEXT NOT NULL DEFAULT '{}',
  value_json TEXT NOT NULL DEFAULT '{}',
  valid_from TEXT,
  valid_to TEXT,
  time_quality TEXT NOT NULL DEFAULT 'unknown',
  document_version_id TEXT REFERENCES document_version(document_version_id),
  mention_id TEXT REFERENCES event_mention(mention_id),
  event_id TEXT REFERENCES event_cluster(cluster_id),
  correction_of TEXT REFERENCES assertion(assertion_id),
  retraction_of TEXT REFERENCES assertion(assertion_id),
  lineage_group TEXT,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','retracted','superseded')),
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assertion_slot ON assertion(slot_key, created_at);
CREATE INDEX IF NOT EXISTS assertion_subject ON assertion(subject_entity_id);

CREATE TABLE IF NOT EXISTS assertion_evidence (
  assertion_id TEXT NOT NULL REFERENCES assertion(assertion_id),
  evidence_no INTEGER NOT NULL,
  document_version_id TEXT,
  quote TEXT,
  url TEXT,
  relation TEXT NOT NULL DEFAULT 'supports' CHECK (relation IN ('supports','contradicts')),
  PRIMARY KEY (assertion_id, evidence_no)
);

CREATE TABLE IF NOT EXISTS slot_selection_history (
  selection_id TEXT PRIMARY KEY,
  slot_key TEXT NOT NULL,
  chosen_assertion_id TEXT REFERENCES assertion(assertion_id),
  disposition TEXT NOT NULL CHECK (disposition IN ('accepted','conflicted','unknown')),
  candidate_ids_json TEXT NOT NULL DEFAULT '[]',
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  sys_from TEXT NOT NULL,
  sys_to TEXT,
  decision_id TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS slot_lookup ON slot_selection_history(slot_key);

-- 语义关系（图）--------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_relation (
  relation_id TEXT PRIMARY KEY,
  from_type TEXT NOT NULL CHECK (from_type IN ('entity','event','process','assertion','evidence')),
  from_id TEXT NOT NULL,
  to_type TEXT NOT NULL CHECK (to_type IN ('entity','event','process','assertion','evidence')),
  to_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (relation IN
    ('participates_in','part_of_process','precedes','follows','supports','contradicts',
     'corrects','retracts','related_to','subsidiary_of','brand_of','refers_to')),
  asserted_by TEXT NOT NULL DEFAULT 'rule' CHECK (asserted_by IN ('rule','manual','model','seed')),
  evidence_json TEXT NOT NULL DEFAULT '[]',
  note TEXT,
  valid_from TEXT,
  valid_to TEXT,
  created_by TEXT,
  created_at TEXT NOT NULL,
  deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS rel_from ON semantic_relation(from_type, from_id, relation);
CREATE INDEX IF NOT EXISTS rel_to ON semantic_relation(to_type, to_id, relation);
-- 单端点索引：不带类型过滤的按 id 查邻接（聚合统计/焦点扩展/实体排序）必须走这两条，
-- 否则前导列 from_type/to_type 未约束会导致 6.5 万行全表扫
CREATE INDEX IF NOT EXISTS rel_from_id ON semantic_relation(from_id);
CREATE INDEX IF NOT EXISTS rel_to_id ON semantic_relation(to_id);

-- 归并决策与流水线 -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS resolution_decision (
  decision_id TEXT PRIMARY KEY,
  mention_id TEXT NOT NULL,
  action TEXT NOT NULL CHECK (action IN
    ('attach','create_provisional','judge_attach','judge_create','pending','manual','reject')),
  target_cluster_id TEXT,
  candidates_json TEXT NOT NULL DEFAULT '[]',
  scores_json TEXT NOT NULL DEFAULT '{}',
  features_json TEXT NOT NULL DEFAULT '{}',
  judge_json TEXT,
  reason_code TEXT,
  model_id TEXT,
  policy_version TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_time ON resolution_decision(created_at);

CREATE TABLE IF NOT EXISTS pipeline_run (
  run_id TEXT PRIMARY KEY,
  stage TEXT NOT NULL,
  input_ref TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','completed','failed','skipped')),
  attempt INTEGER NOT NULL DEFAULT 1,
  detail TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS run_stage ON pipeline_run(stage, status);

CREATE TABLE IF NOT EXISTS outbox (
  outbox_id TEXT PRIMARY KEY,
  aggregate_type TEXT NOT NULL,
  aggregate_id TEXT NOT NULL,
  aggregate_version INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  delivered_at TEXT,
  UNIQUE (aggregate_id, aggregate_version, event_type)
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox(created_at) WHERE delivered_at IS NULL;

CREATE TABLE IF NOT EXISTS projection_state (
  aggregate_type TEXT NOT NULL,
  aggregate_id TEXT NOT NULL,
  requested_version INTEGER NOT NULL,
  indexed_version INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (aggregate_type, aggregate_id)
);

-- 消息总线（Kafka 兼容 topic 的 SQLite 适配）--------------------------------
CREATE TABLE IF NOT EXISTS queue (
  message_id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  trace_id TEXT,
  created_at TEXT NOT NULL,
  available_at TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','running','done','dead')),
  last_error TEXT
);
CREATE INDEX IF NOT EXISTS queue_pick ON queue(topic, status, available_at);

-- 可观测事件流 ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_event (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info' CHECK (level IN ('debug','info','warn','error')),
  stage TEXT NOT NULL,
  trace_id TEXT,
  kind TEXT,
  target_id TEXT,
  message TEXT NOT NULL,
  data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS pe_time ON pipeline_event(event_id);

-- 变化卡与订阅 ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS change_record (
  change_id TEXT PRIMARY KEY,
  slot_key TEXT NOT NULL,
  event_id TEXT,
  subject_entity_id TEXT,
  change_kind TEXT NOT NULL CHECK (change_kind IN
    ('state_transition','correction','conflict','retraction','first_seen','expire')),
  before_json TEXT,
  after_json TEXT,
  importance TEXT NOT NULL DEFAULT 'normal',
  dedupe_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS change_time ON change_record(created_at);

CREATE TABLE IF NOT EXISTS subscription (
  subscription_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  entity_id TEXT REFERENCES entity(entity_id),
  event_types_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delivery (
  delivery_id TEXT PRIMARY KEY,
  subscription_id TEXT NOT NULL REFERENCES subscription(subscription_id),
  change_id TEXT NOT NULL REFERENCES change_record(change_id),
  delivered_at TEXT NOT NULL,
  UNIQUE (subscription_id, change_id)
);

-- 向量（Embedding 投影层，可从文本重建）--------------------------------------
CREATE TABLE IF NOT EXISTS vector (
  vector_id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  owner_type TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  sub_id TEXT NOT NULL DEFAULT '',
  model_id TEXT NOT NULL,
  dim INTEGER NOT NULL,
  text_hash TEXT NOT NULL,
  vec BLOB NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  UNIQUE (role, owner_type, owner_id, sub_id, model_id)
);
CREATE INDEX IF NOT EXISTS vector_role ON vector(role, active);
CREATE INDEX IF NOT EXISTS vector_owner ON vector(owner_type, owner_id, role, active);
"""


def connect(path) -> sqlite3.Connection:
    """打开（必要时创建）数据库连接；WAL + busy_timeout 保证多进程读写安全。"""
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表 + 版本迁移（幂等）。"""
    conn.executescript(SCHEMA_SQL)
    # 迁移：v1 → v2 增加 event_cluster.card_text / card_hash
    cols = [r[1] for r in conn.execute("PRAGMA table_info(event_cluster)").fetchall()]
    if "card_text" not in cols:
        conn.execute("ALTER TABLE event_cluster ADD COLUMN card_text TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE event_cluster ADD COLUMN card_hash TEXT")
    mcols = [r[1] for r in conn.execute("PRAGMA table_info(event_mention)").fetchall()]
    if "event_type_raw" not in mcols:
        conn.execute("ALTER TABLE event_mention ADD COLUMN event_type_raw TEXT")
    # 迁移：semantic_relation 的 CHECK 枚举无法 ALTER，需重建表以纳入新关系类型
    _ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                        "AND name='semantic_relation'").fetchone()
    if _ddl and ("subsidiary_of" not in (_ddl[0] or "") or "'seed'" not in (_ddl[0] or "")):
        conn.execute("ALTER TABLE semantic_relation RENAME TO semantic_relation_old")
        _rel_ddl = SCHEMA_SQL[SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS semantic_relation"):]
        _rel_ddl = _rel_ddl[:_rel_ddl.index("CREATE TABLE IF NOT EXISTS resolution_decision")]
        conn.executescript(_rel_ddl)
        conn.execute("INSERT INTO semantic_relation SELECT * FROM semantic_relation_old")
        conn.execute("DROP TABLE semantic_relation_old")
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


@contextmanager
def tx(conn: sqlite3.Connection):
    """事务助手：BEGIN IMMEDIATE 保证写互斥，异常回滚。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
