#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两代产线的结构化指标拉取（只读，同一脚本跑 v2 旧库与 v3.1 新库，保证口径一致）。

用法：
  python3 compare/line_metrics.py <db_path> <llm_cache_path> <out_json>
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def q1(conn, sql, *a):
    return conn.execute(sql, a).fetchone()[0]


def one(conn, sql, *a):
    r = conn.execute(sql, a).fetchone()
    return r[0] if r else None


def metrics(db_path: str, cache_path: str | None) -> dict:
    # 只读打开：旧库是 v2 终态证据，绝不能写
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    m: dict = {}

    # ---- A. 规模与覆盖 ----
    m["documents"] = q1(conn, "SELECT COUNT(*) FROM document")
    m["document_versions"] = q1(conn, "SELECT COUNT(*) FROM document_version")
    m["duplicates"] = q1(conn, "SELECT COUNT(*) FROM document_version WHERE status='duplicate_of'")
    m["mentions_valid"] = q1(conn, "SELECT COUNT(*) FROM event_mention WHERE status='valid'")
    m["mentions_invalid"] = q1(conn, "SELECT COUNT(*) FROM event_mention WHERE status='invalid'")
    m["mentions_invalid_rate"] = round(m["mentions_invalid"] /
                                       max(m["mentions_valid"] + m["mentions_invalid"], 1), 4)
    m["mentions_per_doc"] = round(m["mentions_valid"] / max(m["documents"], 1), 3)
    m["clusters_active"] = q1(
        conn, "SELECT COUNT(*) FROM event_cluster WHERE deleted_at IS NULL "
              "AND state NOT IN ('redirected','split','deleted')")
    m["clusters_provisional"] = q1(
        conn, "SELECT COUNT(*) FROM event_cluster WHERE deleted_at IS NULL AND state='provisional'")
    m["clusters_resolved"] = q1(
        conn, "SELECT COUNT(*) FROM event_cluster WHERE deleted_at IS NULL AND state='resolved'")
    m["resolved_ratio"] = round(m["clusters_resolved"] / max(m["clusters_active"], 1), 4)
    m["card_coverage"] = round(q1(
        conn, "SELECT COUNT(*) FROM event_cluster WHERE deleted_at IS NULL "
              "AND state NOT IN ('redirected','split','deleted') AND card_text IS NOT NULL "
              "AND card_text<>''") / max(m["clusters_active"], 1), 4)
    m["processes"] = q1(conn, "SELECT COUNT(*) FROM process")
    m["entities"] = q1(conn, "SELECT COUNT(*) FROM entity WHERE deleted_at IS NULL")
    m["entity_aliases"] = q1(conn, "SELECT COUNT(*) FROM entity_alias")
    m["assertions"] = q1(conn, "SELECT COUNT(*) FROM assertion")
    m["selections"] = q1(conn, "SELECT COUNT(*) FROM slot_selection_history")
    m["changes"] = q1(conn, "SELECT COUNT(*) FROM change_record")

    # ---- B. 聚合质量代理 ----
    row = conn.execute("""
        SELECT AVG(n_members), AVG(n_groups), AVG(n_claims)
        FROM (
          SELECT cm.cluster_id,
                 COUNT(*) n_members,
                 COUNT(DISTINCT COALESCE(sl.lineage_group, m.document_version_id)) n_groups,
                 (SELECT SUM(json_array_length(json(m.claims_json)))
                  FROM cluster_membership cm2 JOIN event_mention m2 ON m2.mention_id=cm2.mention_id
                  WHERE cm2.cluster_id=cm.cluster_id AND cm2.removed_at IS NULL
                    AND m2.status='valid') n_claims
          FROM cluster_membership cm
          JOIN event_mention m ON m.mention_id=cm.mention_id AND m.status='valid'
          LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id
          LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id
          WHERE cm.removed_at IS NULL
          GROUP BY cm.cluster_id)""").fetchone()
    m["avg_members_per_cluster"] = round(row[0] or 0, 3)
    m["avg_groups_per_cluster"] = round(row[1] or 0, 3)
    m["avg_claims_per_cluster"] = round(row[2] or 0, 3)
    m["single_member_clusters"] = q1(conn, """
        SELECT COUNT(*) FROM (
          SELECT cm.cluster_id FROM cluster_membership cm
          JOIN event_mention m ON m.mention_id=cm.mention_id AND m.status='valid'
          WHERE cm.removed_at IS NULL GROUP BY cm.cluster_id HAVING COUNT(*)=1)""")
    m["single_member_ratio"] = round(m["single_member_clusters"] /
                                     max(m["clusters_active"], 1), 4)
    m["multi_group_clusters"] = q1(conn, """
        SELECT COUNT(*) FROM (
          SELECT cm.cluster_id FROM cluster_membership cm
          JOIN event_mention m ON m.mention_id=cm.mention_id AND m.status='valid'
          LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id
          LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id
          WHERE cm.removed_at IS NULL
          GROUP BY cm.cluster_id
          HAVING COUNT(DISTINCT COALESCE(sl.lineage_group, m.document_version_id))>=2)""")
    m["multi_group_ratio"] = round(m["multi_group_clusters"] / max(m["clusters_active"], 1), 4)
    # 报道去重压缩比：总报道次数 / 去重后簇数（越高=转载归并越有效）
    m["mentions_to_cluster_ratio"] = round(
        m["mentions_valid"] / max(m["clusters_active"], 1), 3)
    # 平均每提及 claims（抽取深度）
    m["avg_claims_per_mention"] = round(q1(conn, """
        SELECT AVG(json_array_length(json(claims_json))) FROM event_mention
        WHERE status='valid' AND claims_json IS NOT NULL AND claims_json!='[]'""") or 0, 3)

    # ---- C. 决策分布 ----
    m["decisions"] = {r["action"]: r["n"] for r in conn.execute(
        "SELECT action, COUNT(*) n FROM resolution_decision GROUP BY action")}
    tot = sum(m["decisions"].values()) or 1
    m["decisions_ratio"] = {k: round(v / tot, 4) for k, v in m["decisions"].items()}
    # 判别器输出分布（judge_json 里的 decision 字段，v2=8B 输出 / v3.1=Jev 输出）
    try:
        m["judge_decisions"] = {r["d"]: r["n"] for r in conn.execute("""
            SELECT json_extract(judge_json,'$.decision') d, COUNT(*) n
            FROM resolution_decision WHERE judge_json IS NOT NULL GROUP BY d""")}
    except sqlite3.OperationalError:
        m["judge_decisions"] = {}

    # ---- D. 断言质量代理 ----
    m["assertion_status"] = {r["status"]: r["n"] for r in conn.execute(
        "SELECT status, COUNT(*) n FROM assertion GROUP BY status")}
    m["selection_disposition"] = {r["disposition"]: r["n"] for r in conn.execute(
        "SELECT disposition, COUNT(*) n FROM slot_selection_history GROUP BY disposition")}
    asser_tot = m["assertions"] or 1
    m["conflict_rate"] = round(m["selection_disposition"].get("conflicted", 0) /
                               max(m["selections"], 1), 4)

    # ---- E. 图谱 ----
    m["relations"] = {r["relation"]: r["n"] for r in conn.execute(
        "SELECT relation, COUNT(*) n FROM semantic_relation WHERE deleted_at IS NULL "
        "GROUP BY relation")}
    m["relation_total"] = sum(m["relations"].values())
    m["hub_top10_degree"] = [r[0] for r in conn.execute("""
        SELECT cnt FROM (
          SELECT from_id, COUNT(*) cnt FROM semantic_relation
          WHERE relation='participates_in' AND deleted_at IS NULL
          GROUP BY from_id ORDER BY cnt DESC LIMIT 10)""")]

    # ---- F. 向量与模型调用 ----
    try:
        m["vectors_by_role"] = {r["role"]: r["n"] for r in conn.execute(
            "SELECT role, COUNT(*) n FROM vector WHERE active=1 GROUP BY role")}
    except sqlite3.OperationalError:
        m["vectors_by_role"] = {}
    if cache_path and Path(cache_path).exists():
        lc = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        m["cache_models"] = {r[0]: r[1] for r in lc.execute(
            "SELECT model_id, COUNT(*) FROM llm_cache GROUP BY model_id")}
        m["cache_stages"] = {r[0]: r[1] for r in lc.execute(
            "SELECT stage, COUNT(*) FROM llm_cache GROUP BY stage ORDER BY 2 DESC LIMIT 20")}
        lc.close()

    conn.close()
    return m


if __name__ == "__main__":
    db, out = sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "/dev/stdout"
    cache = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    result = metrics(db, cache)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
