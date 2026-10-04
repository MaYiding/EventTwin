# -*- coding: utf-8 -*-
"""流水线总编排：ingest → extract → entities → vectorize → resolve → assertions → project。
（v3.1：向量层随 embedding-8b 回归；判定仍由 Jev 承担。）

可复现设计（04 §11.3）：
- 语料按稳定顺序处理；所有 ID 为 uuid5 确定性派生；
- LLM 输出全部落缓存（data/state/llm_cache.db），重放同一语料 → 逐字一致的图；
- 每个阶段写 pipeline_run（幂等键 = 阶段+输入哈希）与 pipeline_event（可观测）；
- run manifest 记录配置哈希、语料哈希、代码版本。
"""
from __future__ import annotations

import sqlite3

from .. import config as cfg, observe
from ..store import db
from ..util import canonical_json, det_uuid, now_iso, sha256_text
from . import assertions, entities, extract, ingest, project, resolve, vectorize

STAGES = ["ingest", "extract", "entity", "vector", "resolve", "assertion", "project"]


def corpus_hash() -> str:
    cfg.ensure_dirs()
    from pathlib import Path as _P
    dirs = [_P(cfg.ROOT) / d for d in
            cfg.load_config().get("pipeline", {}).get("corpus_dirs", ["data/corpus/incoming"])]
    parts = []
    for d in dirs:
        for fp in sorted(d.glob("*.json")):
            parts.append(fp.name + ":" + sha256_text(fp.read_text(encoding="utf-8"))[:16])
    return sha256_text("|".join(parts))


def reset(conn: sqlite3.Connection, *, hard: bool = False) -> None:
    """清空权威数据（保留 LLM 缓存；hard=True 连缓存一起清）。"""
    import shutil
    conn.executescript("""
    DELETE FROM queue; DELETE FROM outbox; DELETE FROM projection_state;
    DELETE FROM pipeline_event; DELETE FROM pipeline_run;
    DELETE FROM slot_selection_history; DELETE FROM assertion_evidence; DELETE FROM assertion;
    DELETE FROM semantic_relation; DELETE FROM resolution_decision;
    DELETE FROM change_record; DELETE FROM delivery; DELETE FROM subscription;
    DELETE FROM cluster_version; DELETE FROM cluster_membership; DELETE FROM event_cluster;
    DELETE FROM event_mention; DELETE FROM process; DELETE FROM entity_alias; DELETE FROM entity;
    DELETE FROM source_lineage; DELETE FROM document_version; DELETE FROM document;
    DELETE FROM raw_snapshot; DELETE FROM source; DELETE FROM vector; DELETE FROM meta;
    """)
    conn.commit()
    if hard and cfg.LLM_CACHE_PATH.exists():
        cfg.LLM_CACHE_PATH.unlink()
    if cfg.OBJECTS_DIR.exists():
        shutil.rmtree(cfg.OBJECTS_DIR)
        cfg.OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    from ..store import vectors
    vectors.invalidate_cache()
    from . import recall as _recall
    _recall.invalidate_index_caches()


def _stage_run(conn, stage: str, input_ref: str, fn, **kwargs):
    """带幂等记录的阶段执行（同输入哈希重复执行会重跑但结果一致——LLM 缓存保证）。"""
    key = f"{stage}:{sha256_text(input_ref)[:24]}"
    run_id = det_uuid("prun", key)
    conn.execute(
        "INSERT INTO pipeline_run(run_id, stage, input_ref, idempotency_key, status, "
        "started_at) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(idempotency_key) DO UPDATE SET status='running', started_at=excluded.started_at",
        (run_id, stage, input_ref[:200], key, "running", now_iso()))
    conn.commit()
    observe.emit(conn, stage, f"── 阶段开始: {stage} ──", kind="stage.start")
    try:
        out = fn(conn, **kwargs)
    except Exception as e:  # noqa: BLE001
        conn.execute("UPDATE pipeline_run SET status='failed', finished_at=?, detail=? "
                     "WHERE run_id=?", (now_iso(), str(e)[:1000], run_id))
        conn.commit()
        observe.emit(conn, stage, f"阶段失败: {stage}: {e}", level="error", kind="stage.failed")
        raise
    # detail 记录做安全序列化：阶段统计若混入非字符串键（bool 等）不应拖垮已完成的阶段
    try:
        detail = canonical_json(out)[:1000] if out else None
    except (TypeError, ValueError):
        detail = str(out)[:1000]
    conn.execute("UPDATE pipeline_run SET status='completed', finished_at=?, detail=? "
                 "WHERE run_id=?", (now_iso(), detail, run_id))
    conn.commit()
    try:
        observe.emit(conn, stage, f"── 阶段完成: {stage} ──", kind="stage.done", data=out)
    except (TypeError, ValueError):
        observe.emit(conn, stage, f"── 阶段完成: {stage} ──", kind="stage.done")
    return out


def recover_stale_running(conn: sqlite3.Connection) -> int:
    """把上一进程遗留的 running 消息恢复为 pending（单写者批次模型下安全，重跑幂等）。"""
    n = conn.execute("UPDATE queue SET status='pending', available_at=? WHERE status='running'",
                     (now_iso(),)).rowcount
    if n:
        conn.commit()
        observe.emit(conn, "pipeline", f"恢复 {n} 条遗留 running 消息为 pending",
                     level="warn", kind="pipeline.recover")
    return n


def reconcile_topic(conn: sqlite3.Connection, topic: str, stage: str) -> None:
    """批次阶段完成后，把对应 topic 的 pending 消息标记 done（逻辑已由批次处理）。"""
    n = conn.execute("UPDATE queue SET status='done' WHERE topic=? AND status='pending'",
                     (topic,)).rowcount
    if n:
        conn.commit()
        observe.emit(conn, stage, f"对账: {topic} 标记 {n} 条为已处理", kind="pipeline.reconcile")


def run_pipeline(conn: sqlite3.Connection, *, stages: list[str] | None = None,
                 use_cache: bool = True, corpus_dir=None) -> dict:
    """端到端执行；stages 可选子集。返回各阶段统计。"""
    import json as _json
    cfg.ensure_dirs()
    conf = cfg.load_config()
    recover_stale_running(conn)
    manifest = {
        "config_hash": sha256_text(canonical_json(conf))[:16],
        "corpus_hash": corpus_hash()[:16],
        "code_version": _code_version(),
        "llm_cache": bool(conf["pipeline"]["llm_cache"] and use_cache),
        "started_at": now_iso(),
    }
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('last_run_manifest', ?)",
                 (_json.dumps(manifest, ensure_ascii=False),))
    conn.commit()
    observe.emit(conn, "pipeline", "流水线启动", kind="pipeline.start", data=manifest)
    todo = stages or STAGES
    results = {}
    for st in todo:
        if st == "ingest":
            results[st] = _stage_run(conn, "ingest", f"corpus:{corpus_hash()[:16]}",
                                     ingest.run, corpus_dir=corpus_dir)
        elif st == "entity":
            results[st] = _stage_run(conn, "entity", "all", entities.run)
            from .. import hierarchy as _H
            with conn:
                _H.apply_seeds(conn)  # 母子品牌层级边 + 别名种子（小米集团↔小米汽车 等）
        elif st == "vector":
            results[st] = _stage_run(conn, "vector", f"docs:{corpus_hash()[:16]}",
                                     vectorize.run, use_cache=use_cache)
        elif st == "resolve":
            results[st] = _stage_run(conn, "resolve", f"mentions:{corpus_hash()[:16]}",
                                     resolve.run, use_cache=use_cache)
        elif st == "extract":
            results[st] = _stage_run(conn, "extract", f"docs:{corpus_hash()[:16]}",
                                     extract.run, use_cache=use_cache)
            reconcile_topic(conn, "intel.document.parsed.v1", "extract")
        elif st == "assertion":
            results[st] = _stage_run(conn, "assertion", "queue", assertions.run)
        elif st == "project":
            results[st] = _stage_run(conn, "project", "outbox", project.run)
            reconcile_topic(conn, "intel.event.changed.v1", "project")
    observe.emit(conn, "pipeline", "流水线完成", kind="pipeline.done", data=results)
    return results


def _code_version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def stats(conn: sqlite3.Connection) -> dict:
    def one(q, *a):
        return conn.execute(q, a).fetchone()[0]

    return {
        "documents": one("SELECT COUNT(*) FROM document"),
        "document_versions": one("SELECT COUNT(*) FROM document_version"),
        "duplicates": one("SELECT COUNT(*) FROM document_version WHERE status='duplicate_of'"),
        "mentions": one("SELECT COUNT(*) FROM event_mention WHERE status='valid'"),
        "mentions_invalid": one("SELECT COUNT(*) FROM event_mention WHERE status='invalid'"),
        "clusters": one("SELECT COUNT(*) FROM event_cluster WHERE deleted_at IS NULL "
                        "AND state NOT IN ('redirected','split','deleted')"),
        "processes": one("SELECT COUNT(*) FROM process"),
        "entities": one("SELECT COUNT(*) FROM entity WHERE deleted_at IS NULL"),
        "assertions": one("SELECT COUNT(*) FROM assertion"),
        "selections": one("SELECT COUNT(*) FROM slot_selection_history"),
        "relations": one("SELECT COUNT(*) FROM semantic_relation WHERE deleted_at IS NULL"),
        "changes": one("SELECT COUNT(*) FROM change_record"),
        "vectors": one("SELECT COUNT(*) FROM vector WHERE active=1"),
        "decisions": {r["action"]: r["n"] for r in conn.execute(
            "SELECT action, COUNT(*) n FROM resolution_decision GROUP BY action")},
        "pending_mentions": one(
            "SELECT COUNT(*) FROM event_mention m WHERE m.status='valid' AND NOT EXISTS ("
            "SELECT 1 FROM cluster_membership cm WHERE cm.mention_id=m.mention_id "
            "AND cm.removed_at IS NULL)"),
    }
