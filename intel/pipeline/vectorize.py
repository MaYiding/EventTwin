# -*- coding: utf-8 -*-
"""向量化阶段：提及框架 / 证据句 / 文档 chunk 三类表示（04 §6.2）。

事件归类用规范框架文本（避免长文背景稀释动作）；
问答用摘要与原文 chunk。两种用途各自锁定输入模板。
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from .. import config as cfg, llm, observe
from ..store import vectors
from ..util import sha256_text

CHUNK = 1200


def run(conn: sqlite3.Connection, *, use_cache: bool = True) -> dict:
    conf = cfg.load_config()
    model_id = conf["models"]["embed_model"]
    dim = conf["models"]["embed_dimensions"]
    stats = {"frames": 0, "evidence": 0, "chunks": 0}

    # 1) 提及框架 + 证据句（未向量化 / 框架文本已变化的）
    rows = conn.execute(
        "SELECT mention_id, frame_text, evidence_json FROM event_mention "
        "WHERE status='valid' ORDER BY created_at, mention_id").fetchall()
    # 批量载入已有向量指纹（集合比对，O(N)——逐条查询在 2 万提及下是 O(N²) 天级瓶颈）
    existing_frames = {r["owner_id"]: r["text_hash"] for r in conn.execute(
        "SELECT owner_id, text_hash FROM vector WHERE role='mention_frame' "
        "AND owner_type='mention' AND active=1")}
    todo = []
    for r in rows:
        h = sha256_text(r["frame_text"])
        if existing_frames.get(r["mention_id"]) == h:
            continue
        todo.append(r)
    if todo:
        texts = [r["frame_text"] for r in todo]
        mat = llm.embed(texts, stage="embed.frame")
        for r, vec in zip(todo, mat):
            vectors.add_vector(conn, role="mention_frame", owner_type="mention",
                               owner_id=r["mention_id"], sub_id="", model_id=model_id,
                               dim=dim, vec=vec, text_hash=sha256_text(r["frame_text"]))
            stats["frames"] += 1
        conn.commit()

    # 证据句（每提及最多 3 条，控制向量数）
    ev_todo = []
    for r in todo:
        evs = json.loads(r["evidence_json"] or "[]")[:3]
        for i, ev in enumerate(evs):
            q = (ev.get("quote") or "").strip()
            if len(q) >= 8:
                ev_todo.append((r["mention_id"], f"ev{i}", q))
    if ev_todo:
        mat = llm.embed([q for _, _, q in ev_todo], stage="embed.evidence")
        for (mid, sub, q), vec in zip(ev_todo, mat):
            vectors.add_vector(conn, role="evidence", owner_type="mention", owner_id=mid,
                               sub_id=sub, model_id=model_id, dim=dim, vec=vec,
                               text_hash=sha256_text(q))
            stats["evidence"] += 1
        conn.commit()

    # 2) 文档 chunk（问答 dense 通道）
    drows = conn.execute(
        "SELECT document_version_id, normalized_text FROM document_version "
        "WHERE status IN ('extracted','resolved') AND normalized_text IS NOT NULL "
        "ORDER BY created_at, document_version_id").fetchall()
    existing_chunks = {(r["owner_id"], r["sub_id"]): r["text_hash"] for r in conn.execute(
        "SELECT owner_id, sub_id, text_hash FROM vector WHERE role='doc_chunk' "
        "AND owner_type='docver' AND active=1")}
    chunk_todo = []
    for r in drows:
        text = r["normalized_text"]
        for i in range(0, len(text), CHUNK):
            c = text[i:i + CHUNK]
            if len(c) < 50:
                continue
            if existing_chunks.get((r["document_version_id"], str(i))) == sha256_text(c):
                continue
            chunk_todo.append((r["document_version_id"], str(i), c))
    # 限量防失控：单次最多 400 个 chunk
    chunk_todo = chunk_todo[:400]
    B = 32
    for i in range(0, len(chunk_todo), B):
        batch = chunk_todo[i:i + B]
        mat = llm.embed([c for _, _, c in batch], stage="embed.chunk")
        for (docv, sub, c), vec in zip(batch, mat):
            vectors.add_vector(conn, role="doc_chunk", owner_type="docver", owner_id=docv,
                               sub_id=sub, model_id=model_id, dim=dim, vec=vec,
                               text_hash=sha256_text(c))
            stats["chunks"] += 1
        conn.commit()

    vectors.invalidate_cache()
    observe.emit(conn, "vector", f"向量化完成: {stats}", kind="vector.stage_done", data=stats)
    return stats
