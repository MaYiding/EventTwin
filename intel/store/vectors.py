# -*- coding: utf-8 -*-
"""向量投影层：保存 Embedding 并提供暴力余弦 ANN（OpenSearch 的单机映射）。

铁律对齐：向量是可重建的派生数据，不是权威事实；每次检索按簇去重；
模型/维度记录在每行上，不同空间不混比。
"""
from __future__ import annotations

import sqlite3
import threading

import numpy as np

from ..util import det_uuid, now_iso

# 进程内缓存：role → (矩阵, owner 列表, 加载时的表行数水位)
_cache: dict = {}
_cache_lock = threading.Lock()


def add_vector(conn: sqlite3.Connection, *, role: str, owner_type: str, owner_id: str,
               sub_id: str, model_id: str, dim: int, vec: np.ndarray,
               text_hash: str) -> str:
    assert vec.shape == (dim,), f"向量维度不符: {vec.shape} != {dim}"
    vector_id = det_uuid("vec", role, owner_type, owner_id, sub_id, model_id)
    conn.execute(
        "INSERT INTO vector(vector_id, role, owner_type, owner_id, sub_id, model_id, dim, "
        "text_hash, vec, active, created_at) VALUES (?,?,?,?,?,?,?,?,?,1,?) "
        "ON CONFLICT(role, owner_type, owner_id, sub_id, model_id) DO UPDATE SET "
        "vec=excluded.vec, text_hash=excluded.text_hash, active=1, dim=excluded.dim",
        (vector_id, role, owner_type, owner_id, sub_id, model_id, dim,
         text_hash, np.asarray(vec, dtype=np.float32).tobytes(), now_iso()),
    )
    return vector_id


def deactivate(conn: sqlite3.Connection, *, owner_type: str, owner_id: str,
               role: str | None = None) -> None:
    q = "UPDATE vector SET active=0 WHERE owner_type=? AND owner_id=?"
    args: list = [owner_type, owner_id]
    if role:
        q += " AND role=?"
        args.append(role)
    conn.execute(q, args)


def _load_role(conn: sqlite3.Connection, role: str):
    """加载某 role 的全部活跃向量到内存（带行数水位失效）。"""
    with _cache_lock:
        total = conn.execute("SELECT COUNT(*) n FROM vector WHERE active=1").fetchone()["n"]
        cached = _cache.get(role)
        if cached is not None and cached["total"] == total:
            return cached
        rows = conn.execute(
            "SELECT vector_id, owner_type, owner_id, sub_id, dim, vec FROM vector "
            "WHERE active=1 AND role=?",
            (role,),
        ).fetchall()
        if not rows:
            entry = {"mat": np.zeros((0, 1), dtype=np.float32), "owners": [], "total": total}
            _cache[role] = entry
            return entry
        dim = rows[0]["dim"]
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms  # L2 归一化后点积即余弦
        owners = [(r["owner_type"], r["owner_id"], r["sub_id"]) for r in rows]
        entry = {"mat": mat, "owners": owners, "total": total}
        _cache[role] = entry
        return entry


def cosine_search(conn: sqlite3.Connection, query_vec: np.ndarray, role: str,
                  top_k: int) -> list[tuple[str, str, str, float]]:
    """余弦近邻：返回 [(owner_type, owner_id, sub_id, score)] 降序。"""
    q = np.asarray(query_vec, dtype=np.float32)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    entry = _load_role(conn, role)
    if entry["mat"].shape[0] == 0:
        return []
    sims = entry["mat"] @ q
    k = min(top_k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(entry["owners"][i][0], entry["owners"][i][1], entry["owners"][i][2], float(sims[i]))
            for i in idx]


def get_vector(conn: sqlite3.Connection, *, owner_type: str, owner_id: str,
               role: str) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vec FROM vector WHERE owner_type=? AND owner_id=? AND role=? AND active=1 "
        "ORDER BY created_at DESC LIMIT 1",
        (owner_type, owner_id, role),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row["vec"], dtype=np.float32).copy()


def invalidate_cache() -> None:
    with _cache_lock:
        _cache.clear()
