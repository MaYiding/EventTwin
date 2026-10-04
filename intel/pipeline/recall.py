# -*- coding: utf-8 -*-
"""多路候选召回（04 §7）：ANN(按簇去重) + BM25 + 结构化 + 未投影补丁 + 越窗兜底。

v3.1（embedding-8b 回归后）：
- ANN 通道恢复（簇质心 + 代表证据余弦，qwen3-embedding-8b 4096 维）；
- S_semantic 特征用回质心余弦（resolve._rank_candidates）；
- 同事件概率仍由 Jev 判定（resolve._jev_psame）——embedding 只做召回，不参与判定。

铁律：
- 召回是起点不是保证；只对明确不相容条件做硬过滤（唯一编号不同 / SKU 不同 /
  市场不同），未知字段不排除；
- 记录召回健康度 coverage_ok；索引不健康时低相似度不支持自动新建。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .. import config as cfg
from ..nlp import BM25Index
from ..store import vectors
from ..util import CST, parse_datetime

_idx_cache: dict = {}


def cluster_bm25_index(conn: sqlite3.Connection) -> BM25Index:
    """事件簇 BM25 索引（summary+框架文本；派生层，随时可从 DB 重建）。"""
    key = _cluster_version_watermark(conn)
    if _idx_cache.get("key") == key:
        return _idx_cache["idx"]
    idx = BM25Index()
    rows = conn.execute(
        "SELECT cluster_id, summary, frame_json FROM event_cluster "
        "WHERE deleted_at IS NULL AND state NOT IN ('redirected','split','deleted')").fetchall()
    for r in rows:
        frame = json.loads(r["frame_json"] or "{}")
        text = f"{r['summary']} {frame.get('frame_text', '')}"
        idx.add(r["cluster_id"], text)
    _idx_cache.update({"key": key, "idx": idx})
    return idx


def _cluster_version_watermark(conn) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(version), 0) s, COUNT(*) n FROM event_cluster "
        "WHERE deleted_at IS NULL").fetchone()
    return row["s"] * 1000 + row["n"]


def invalidate_index_caches() -> None:
    """投影水位推进后失效进程内派生索引缓存（project 阶段调用）。"""
    _idx_cache.pop("key", None)


def cluster_lexical_scores(conn: sqlite3.Connection, query: str) -> dict[str, float]:
    """BM25 词面分：cluster_id → score（仅 >0 项）。供排序阶段作 S_semantic 特征。"""
    return dict(cluster_bm25_index(conn).score(query))


def recall_candidates(conn, mention: sqlite3.Row, query_vec=None, *,
                      expand: bool = False, allowed_ids: set | None = None) -> dict:
    """返回 {candidates: [cluster_id...], channels: {...}, coverage_ok, excluded: [...]}

    query_vec：新提及框架向量（ANN 通道用；None 时 ANN 通道降级为空，不阻断）。
    allowed_ids：时间片快照候选集（并行判别时的确定性边界）——
    片内计划只能看到片开始前已存在的簇，片间屏障保证重放一致（04 §10.2 串行键的时间片化）。
    """
    conf = cfg.load_config()["retrieval"]
    _allow = (lambda cid: True) if allowed_ids is None else (lambda cid: cid in allowed_ids)
    limit = conf["expanded_pool_limit"] if expand else conf["candidate_pool_limit"]
    channels: dict[str, list[str]] = {}
    excluded: list[dict] = []

    ann_ids: list[str] = []
    if query_vec is not None:
        try:
            hits = vectors.cosine_search(conn, query_vec, "cluster_centroid",
                                         top_k=conf["ann_initial_vector_hits"])
            ann_ids = [oid for _t, oid, _s, _sc in hits]
            hits_rep = vectors.cosine_search(conn, query_vec, "cluster_rep",
                                             top_k=conf["ann_initial_vector_hits"] // 2)
            for _t, oid, _s, _sc in hits_rep:
                if oid not in ann_ids:
                    ann_ids.append(oid)
        except Exception:  # noqa: BLE001 ANN 故障：降级，不把召回失败当新事件
            ann_ids = []
    channels["ann"] = ann_ids

    try:
        bm25_ids = cluster_bm25_index(conn).top(mention["frame_text"], conf["bm25_cluster_limit"])
    except Exception:  # noqa: BLE001
        bm25_ids = []
    channels["bm25"] = bm25_ids

    # 结构化通道（S1 修复：复合 blocking key——hub 实体（参与数百簇）按行序截断会
    # 把该并的候选簇挤出可见范围，是过分裂根因 R1 的 42% 的主要机理）：
    # 同实体 × 同类型 × 事件时间窗相容 先过滤，再按共享实体数降序 + cluster_id 定序截断
    ent_ids = _mention_entity_ids(mention)
    structured: list[str] = []
    if ent_ids:
        ph = ",".join("?" for _ in ent_ids)
        limit = conf.get("structured_cluster_limit", 50)
        rows = conn.execute(
            f"""SELECT r.to_id cid, COUNT(DISTINCT r.from_id) n_ent
                FROM semantic_relation r
                JOIN event_cluster k ON k.cluster_id=r.to_id
                     AND k.deleted_at IS NULL AND k.state NOT IN ('redirected','split','deleted')
                WHERE r.relation='participates_in' AND r.from_type='entity'
                  AND r.from_id IN ({ph}) AND r.deleted_at IS NULL
                  AND k.event_type = ?
                  AND COALESCE(k.event_time_lower,'9999') <= COALESCE(?, '9999')
                  AND COALESCE(k.event_time_upper, k.event_time_lower, '0000') >= COALESCE(?, '0000')
                GROUP BY r.to_id
                ORDER BY n_ent DESC, cid LIMIT {int(limit) * 2}""",
            (*ent_ids, mention["event_type"],
             mention["event_time_upper"] or "9999",
             mention["event_time_lower"] or "0000")).fetchall()
        # 时间窗相容：簇时间与提及时间有交集（上界>=提及下界 且 下界<=提及上界）；
        # 未命中时间相容的回退到无时间条件（不硬杀——召回宁多勿漏，硬过滤另有其职）
        structured = [r["cid"] for r in rows]
        if not structured:
            rows = conn.execute(
                f"""SELECT r.to_id cid, COUNT(DISTINCT r.from_id) n_ent
                    FROM semantic_relation r
                    JOIN event_cluster k ON k.cluster_id=r.to_id
                         AND k.deleted_at IS NULL AND k.state NOT IN ('redirected','split','deleted')
                    WHERE r.relation='participates_in' AND r.from_type='entity'
                      AND r.from_id IN ({ph}) AND r.deleted_at IS NULL
                    GROUP BY r.to_id ORDER BY n_ent DESC, cid LIMIT {int(limit)}""",
                (*ent_ids,)).fetchall()
            structured = [r["cid"] for r in rows]
    channels["structured"] = structured

    # 未投影补丁：事务已提交但投影版本落后的簇（消除索引延迟导致的伪新建）
    unindexed = conn.execute(
        "SELECT c.cluster_id FROM event_cluster c LEFT JOIN projection_state p "
        "ON p.aggregate_type='event_cluster' AND p.aggregate_id=c.cluster_id "
        "WHERE c.deleted_at IS NULL AND c.state NOT IN ('redirected','split','deleted') "
        "AND (p.indexed_version IS NULL OR p.indexed_version < c.version) LIMIT 50").fetchall()
    channels["unindexed_patch"] = [r["cluster_id"] for r in unindexed]

    # 合并去重（通道优先级 + cluster_id 双键确定性排序后截断）：
    # 通道内部顺序在检索分数并列时不定，直接按出现序截断会让保留的候选集合随行序漂移
    seen: dict[str, int] = {}
    for pri, ch in enumerate(("ann", "bm25", "structured", "unindexed_patch")):
        for cid in channels[ch]:
            if _allow(cid) and cid not in seen:
                seen[cid] = pri
    pool = [cid for cid, _ in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))[:limit]]

    coverage_ok = bool(pool) and (bool(ann_ids) or bool(bm25_ids) or bool(structured))
    # 各通道全空且库里已有簇 → 覆盖异常（除非库本身为空）
    n_clusters = conn.execute(
        "SELECT COUNT(*) n FROM event_cluster WHERE deleted_at IS NULL "
        "AND state NOT IN ('redirected','split','deleted')").fetchone()["n"]
    if n_clusters > 0 and not (ann_ids or bm25_ids or structured or unindexed):
        coverage_ok = False

    # 硬过滤：仅明确不相容条件
    filtered = []
    for cid in pool:
        reason = _hard_exclusion(conn, mention, cid)
        if reason:
            excluded.append({"cluster_id": cid, "reason": reason})
        else:
            filtered.append(cid)
    # 越窗兜底：过滤后为空但原本有候选 → 回退不做时间硬过滤
    if not filtered and excluded and not expand:
        filtered = [e["cluster_id"] for e in excluded
                    if not str(e["reason"]).startswith("time_gap") and _allow(e["cluster_id"])]
        if filtered:
            channels["fallback_window"] = filtered
    # 排除列表按 cluster_id 定序：池序依赖召回通道行序，直接落库会造成 decision 留痕漂移
    excluded.sort(key=lambda e: e["cluster_id"])
    return {"candidates": filtered, "channels": channels, "coverage_ok": coverage_ok,
            "excluded": excluded, "pool_before_filter": pool}


def _mention_entity_ids(mention) -> list[str]:
    ids = []
    for field in ("actor_json", "object_json"):
        try:
            arr = json.loads(mention[field] or "[]")
        except json.JSONDecodeError:
            arr = []
        for a in arr:
            if isinstance(a, dict) and a.get("entity_id"):
                ids.append(a["entity_id"])
    return ids


def _hard_exclusion(conn, mention, cluster_id: str) -> str | None:
    """只排除明确不相容的候选；返回原因码，None 表示保留。"""
    cluster = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cluster_id,)).fetchone()
    if cluster is None or cluster["deleted_at"]:
        return "cluster_missing"
    m_scope = json.loads(mention["scope_json"] or "{}")
    c_frame = json.loads(cluster["frame_json"] or "{}")
    c_scope = c_frame.get("scope") or {}

    # 唯一业务编号明确不同（公告号/专利号等）
    mb, cb = m_scope.get("business_no"), c_scope.get("business_no")
    if mb and cb and str(mb).strip() != str(cb).strip():
        return "distinct_business_no"
    # SKU 明确不同
    ms, cs = m_scope.get("sku"), c_scope.get("sku")
    if ms and cs and _norm(ms) != _norm(cs):
        return "distinct_sku"
    # 市场明确不同（价格类事件）
    mk, ck = m_scope.get("market"), c_scope.get("market")
    if mk and ck and str(mk).strip().upper() != str(ck).strip().upper():
        return "distinct_market"
    # 并购类：阶段互斥（宣布≠交割）
    if mention["event_type"] == cluster["event_type"] == "merger_deal":
        mp, cp = mention["event_phase"], cluster["frame_json"] and c_frame.get("phase")
        if mp and cp and {mp, cp} <= {"announced", "planned", "completed", "effective"} and mp != cp:
            return "incompatible_phase"
    # 时间窗：两边时间都明确且相距超过类型阈值（未知不排除）
    gap_conf = cfg.load_config()["cluster"]["hard_time_gap_days"]
    days = gap_conf.get(mention["event_type"] or "default", gap_conf.get("default", 120))
    m_lo = _dt(mention["event_time_lower"])
    c_lo = _dt(cluster["event_time_lower"])
    c_hi = _dt(cluster["event_time_upper"]) or c_lo
    if m_lo and c_lo:
        m_hi = _dt(mention["event_time_upper"]) or m_lo
        if m_lo > c_hi:
            gap = (m_lo - c_hi).days
        elif c_lo > m_hi:
            gap = (c_lo - m_hi).days
        else:
            gap = 0
        if gap > days:
            return f"time_gap:{gap}d>{days}d"
    return None


def _norm(s: str) -> str:
    return "".join(str(s).lower().split())


def _dt(iso_str: str | None):
    if not iso_str:
        return None
    dt, _ = parse_datetime(iso_str)
    if dt is None:
        try:
            dt = datetime.fromisoformat(iso_str)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CST)
    return dt
