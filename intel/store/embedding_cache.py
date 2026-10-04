# -*- coding: utf-8 -*-
"""Embedding 独立缓存（DATA_DESIGN.md L5 层）。

幂等键 = (model_id, text_hash)：同文本同模型永不重复向量化。
独立于 llm_cache.db（chat/rerank）——可单独清理、重建、审计；
首轮迁移来源为 intel.db 的 vector 表（已含 text_hash + vec）。
"""
from __future__ import annotations

import sqlite3
import threading

import numpy as np

from .. import config as cfg
from ..util import now_iso, sha256_text

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    cfg.ensure_dirs()
    conn = sqlite3.connect(str(cfg.EMBED_CACHE_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embedding_cache (
      model_id  TEXT NOT NULL,
      text_hash TEXT NOT NULL,
      dim       INTEGER NOT NULL,
      vec       BLOB NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY (model_id, text_hash)
    )""")
    conn.commit()
    return conn


def key_for(model_id: str, text: str) -> str:
    return sha256_text(text)


def get(model_id: str, text: str) -> np.ndarray | None:
    """命中返回已归一化向量，未命中返回 None。"""
    th = sha256_text(text)
    try:
        conn = _conn()
        row = conn.execute("SELECT vec FROM embedding_cache WHERE model_id=? AND text_hash=?",
                           (model_id, th)).fetchone()
        conn.close()
        if row is None:
            return None
        vec = np.frombuffer(row[0], dtype=np.float32).copy()
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    except sqlite3.Error:
        return None


def put(model_id: str, text: str, vec: np.ndarray) -> None:
    th = sha256_text(text)
    with _lock:
        try:
            conn = _conn()
            conn.execute(
                "INSERT OR IGNORE INTO embedding_cache(model_id, text_hash, dim, vec, created_at) "
                "VALUES (?,?,?,?,?)",
                (model_id, th, int(vec.shape[0]),
                 np.asarray(vec, dtype=np.float32).tobytes(), now_iso()))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass


def migrate_from_vector_table() -> int:
    """一次性迁移：intel.db vector 表 → embedding_cache.db（text_hash+model+vec）。"""
    conn_main = sqlite3.connect(str(cfg.DB_PATH))
    rows = conn_main.execute(
        "SELECT model_id, text_hash, dim, vec FROM vector").fetchall()
    conn_main.close()
    n = 0
    out = _conn()
    for model_id, text_hash, dim, vec in rows:
        out.execute(
            "INSERT OR IGNORE INTO embedding_cache(model_id, text_hash, dim, vec, created_at) "
            "VALUES (?,?,?,?,?)", (model_id, text_hash, dim, vec, now_iso()))
        n += 1
    out.commit()
    out.close()
    return n


def stats() -> dict:
    try:
        conn = _conn()
        r = conn.execute("SELECT model_id, COUNT(*) n, SUM(dim) dims FROM embedding_cache "
                         "GROUP BY model_id").fetchall()
        conn.close()
        return {row[0]: {"vectors": row[1]} for row in r}
    except sqlite3.Error:
        return {}
