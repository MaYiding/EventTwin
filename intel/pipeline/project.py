# -*- coding: utf-8 -*-
"""异步投影（04 §10.4）：消费 outbox → 更新 projection_state 水位。

BM25 是进程内重建的派生索引（对齐"OpenSearch 索引可重建、无独立权威事实"），
这里的水位记录用于：召回的"未投影补丁"通道 + API 侧发现索引落后时回查事件头。
"""
from __future__ import annotations

import json
import sqlite3

from .. import observe
from ..util import now_iso


def run(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT * FROM outbox WHERE delivered_at IS NULL ORDER BY created_at, outbox_id "
        "LIMIT 500").fetchall()
    n = 0
    for r in rows:
        payload = json.loads(r["payload_json"])
        conn.execute(
            "INSERT INTO projection_state(aggregate_type, aggregate_id, requested_version, "
            "indexed_version, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(aggregate_type, aggregate_id) DO UPDATE SET "
            "indexed_version=MAX(indexed_version, excluded.indexed_version), "
            "requested_version=MAX(requested_version, excluded.requested_version), "
            "updated_at=excluded.updated_at",
            (r["aggregate_type"], r["aggregate_id"], r["aggregate_version"],
             r["aggregate_version"], now_iso()))
        conn.execute("UPDATE outbox SET delivered_at=? WHERE outbox_id=?",
                     (now_iso(), r["outbox_id"]))
        n += 1
        if payload.get("version"):
            observe.emit(conn, "project",
                         f"投影: 事件 {payload.get('cluster_id', '')[:12]} → v{payload['version']}",
                         kind="project.indexed", target_id=payload.get("cluster_id"))
    from . import recall as _recall
    _recall.invalidate_index_caches()  # BM25 索引缓存随水位失效（派生层可重建）
    if n:
        observe.emit(conn, "project", f"投影阶段完成: {n} 条", kind="project.stage_done",
                     data={"delivered": n})
    return {"delivered": n}
