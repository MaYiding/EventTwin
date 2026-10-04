#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性补齐缺失/过期的事件卡向量（可并发于流水线安全运行，WAL）。

口径：对每个活跃簇，若无 active 的 event_card 向量、或 text_hash ≠ card_hash，
则重嵌（走 embedding 缓存，同文本零成本）。幂等。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3  # noqa: E402

from intel import config as cfg, llm  # noqa: E402
from intel.store import db, vectors  # noqa: E402


def main() -> int:
    conn = db.connect(cfg.DB_PATH)
    db.init_db(conn)
    conf = cfg.load_config()
    model_id, dim = conf["models"]["embed_model"], conf["models"]["embed_dimensions"]

    rows = conn.execute("""
        SELECT c.cluster_id, c.card_text, c.card_hash,
               (SELECT v.text_hash FROM vector v WHERE v.role='event_card'
                 AND v.owner_type='cluster' AND v.owner_id=c.cluster_id AND v.active=1
                 LIMIT 1) AS vec_hash
        FROM event_cluster c
        WHERE c.deleted_at IS NULL
          AND c.state NOT IN ('redirected','split','deleted')
          AND c.card_text <> ''
        ORDER BY c.cluster_id""").fetchall()
    todo = [r for r in rows if r["vec_hash"] != r["card_hash"]]
    print(f"活跃簇 {len(rows)}，需补齐卡片向量 {len(todo)}")
    n = 0
    B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        try:
            mat = llm.embed([r["card_text"] for r in batch], stage="embed.event_card")
        except Exception as e:  # noqa: BLE001
            print(f"批次 {i} 失败: {e}")
            continue
        for r, vec in zip(batch, mat):
            vectors.add_vector(conn, role="event_card", owner_type="cluster",
                               owner_id=r["cluster_id"], sub_id="", model_id=model_id,
                               dim=dim, vec=vec, text_hash=r["card_hash"])
            n += 1
        conn.commit()
        if (i // B) % 10 == 0:
            print(f"  进度 {i + len(batch)}/{len(todo)}")
    conn.close()
    print(f"补齐完成: {n} 个卡片向量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
