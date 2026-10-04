# -*- coding: utf-8 -*-
"""统一可观测性：pipeline_event 事件流（OTel 的单机映射）。

trace_id 贯穿 抓取→抽取→召回→判别→事实选择→生成；
SSE 接口按 event_id 增量轮询本表，实现 Web 实时观测。
"""
from __future__ import annotations

import sqlite3

from .util import now_iso

# 模块级当前 trace（进程内流水线串行执行，够用；多线程阶段各自传参）
CURRENT_TRACE: str | None = None


def emit(conn: sqlite3.Connection, stage: str, message: str, *,
         level: str = "info", kind: str | None = None, target_id: str | None = None,
         trace_id: str | None = None, data: dict | None = None) -> int:
    import json
    cur = conn.execute(
        "INSERT INTO pipeline_event(ts, level, stage, trace_id, kind, target_id, message, data_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (now_iso(), level, stage, trace_id or CURRENT_TRACE, kind, target_id, message,
         json.dumps(data or {}, ensure_ascii=False)),
    )
    conn.commit()
    return cur.lastrowid


def recent(conn: sqlite3.Connection, *, after_id: int = 0, limit: int = 200,
           stage: str | None = None, level: str | None = None,
           kind: str | None = None) -> list[dict]:
    q = ("SELECT * FROM pipeline_event WHERE event_id>? ")
    args: list = [after_id]
    if stage:
        q += "AND stage=? "
        args.append(stage)
    if level:
        q += "AND level=? "
        args.append(level)
    if kind:
        q += "AND kind=? "
        args.append(kind)
    q += "ORDER BY event_id DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(q, args).fetchall()]
