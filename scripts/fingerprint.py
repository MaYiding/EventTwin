#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据库内容指纹：剔除天然随运行变化的运行时间戳字段后哈希。

用于验证「--reset 重放可复现」：
    python3 scripts/run_pipeline.py --reset
    python3 scripts/fingerprint.py data/state/intel.db   # 第一次
    python3 scripts/run_pipeline.py --reset
    python3 scripts/fingerprint.py data/state/intel.db   # 第二次，应完全一致
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys

TABLES = [
    ("document", "document_id"), ("document_version", "document_version_id"),
    ("event_mention", "mention_id"), ("event_cluster", "cluster_id"),
    ("cluster_membership", "membership_id"), ("cluster_version", "cluster_id, version"),
    ("process", "process_id"), ("entity", "entity_id"),
    ("assertion", "assertion_id"), ("slot_selection_history", "selection_id"),
    ("semantic_relation", "relation_id"), ("resolution_decision", "decision_id"),
    ("change_record", "change_id"),
]
# 运行时间戳（created_at/sys_from 等）每次重放必然不同，剔除后再比；
# 其余全部字段（含业务时间、决策内容、判别输出）都参与指纹。
SKIP = {"created_at", "updated_at", "added_at", "removed_at", "recorded_at",
        "sys_from", "sys_to", "first_seen", "last_seen", "started_at",
        "finished_at", "fetched_at"}


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "data/state/intel.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    parts = []
    for table, order in TABLES:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
        sig = [{k: r[k] for k in r.keys() if k not in SKIP} for r in rows]
        digest = hashlib.sha256(
            json.dumps(sig, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        parts.append(f"{table}:{len(rows)}:{digest}")
        print(parts[-1])
    print("TOTAL", hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
