# -*- coding: utf-8 -*-
"""持久化消息队列（Kafka 兼容总线的 SQLite 单机映射）。

Topic 划分对齐架构文档 04 §10.5：
  intel.document.ready.v1 / intel.document.parsed.v1 / intel.mention.ready.v1 /
  intel.event.changed.v1 / intel.assertion.candidate.v1 /
  intel.assertion.changed.v1 / intel.retry.v1 / intel.deadletter.v1
至少一次投递 + 消费端幂等；失败指数退避，超次数进死信。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ..util import det_uuid, iso, now_iso, now_ts, CST

TOPICS = {
    "intel.document.ready.v1": "原文已保存，等待解析",
    "intel.document.parsed.v1": "正文已解析，等待抽取",
    "intel.mention.ready.v1": "提及已校验，等待归类",
    "intel.event.changed.v1": "事件成员/版本改变",
    "intel.assertion.candidate.v1": "状态断言候选",
    "intel.assertion.changed.v1": "断言/选择更新",
    "intel.retry.v1": "可恢复错误",
    "intel.deadletter.v1": "超过重试预算",
}


def enqueue(conn: sqlite3.Connection, topic: str, payload: dict,
            trace_id: str | None = None, delay_seconds: float = 0.0) -> str:
    assert topic in TOPICS, f"未知 topic: {topic}"
    message_id = det_uuid("msg", topic, payload.get("idempotency_key") or "",
                          payload.get("document_version_id") or "", payload.get("mention_id") or "",
                          str(len(str(payload))))
    available = iso(datetime.fromtimestamp(now_ts() + delay_seconds, CST))
    conn.execute(
        "INSERT OR IGNORE INTO queue(message_id, topic, payload_json, trace_id, created_at, available_at) "
        "VALUES (?,?,?,?,?,?)",
        (message_id, topic, _dumps(payload), trace_id, now_iso(), available),
    )
    return message_id


def claim(conn: sqlite3.Connection, topic: str) -> sqlite3.Row | None:
    """取一条待处理消息（FIFO），标记 running。"""
    with conn:
        row = conn.execute(
            "SELECT * FROM queue WHERE topic=? AND status='pending' AND available_at<=? "
            "ORDER BY message_id LIMIT 1",
            (topic, now_iso()),
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE queue SET status='running', attempts=attempts+1 WHERE message_id=?",
                     (row["message_id"],))
    return row


def complete(conn: sqlite3.Connection, message_id: str) -> None:
    conn.execute("UPDATE queue SET status='done' WHERE message_id=?", (message_id,))
    conn.commit()


def fail(conn: sqlite3.Connection, message_id: str, error: str) -> str:
    """失败：退避重试或进死信，返回新状态。"""
    row = conn.execute("SELECT attempts, max_attempts FROM queue WHERE message_id=?",
                       (message_id,)).fetchone()
    if row is None:
        return "missing"
    if row["attempts"] >= row["max_attempts"]:
        conn.execute("UPDATE queue SET status='dead', last_error=? WHERE message_id=?",
                     (error[:2000], message_id))
        conn.execute(
            "INSERT OR IGNORE INTO queue(message_id, topic, payload_json, created_at, available_at) "
            "VALUES (?,?,?,?,?)",
            (det_uuid("dead", message_id), "intel.deadletter.v1", _dumps(
                {"source_message_id": message_id, "error": error[:2000]}), now_iso(), now_iso()),
        )
        conn.commit()
        return "dead"
    # 指数退避：2^attempts 秒
    delay = 2 ** min(row["attempts"], 6)
    available = iso(datetime.fromtimestamp(now_ts() + delay, CST))
    conn.execute("UPDATE queue SET status='pending', available_at=?, last_error=? WHERE message_id=?",
                 (available, error[:2000], message_id))
    conn.commit()
    return "retry"


def depths(conn: sqlite3.Connection) -> dict:
    """各 topic 深度（pending/running/dead），用于可观测。"""
    rows = conn.execute(
        "SELECT topic, status, COUNT(*) n FROM queue GROUP BY topic, status"
    ).fetchall()
    out: dict = {}
    for r in rows:
        t = out.setdefault(r["topic"], {"pending": 0, "running": 0, "done": 0, "dead": 0})
        t[r["status"]] = r["n"]
    return out


def _dumps(payload: dict) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
