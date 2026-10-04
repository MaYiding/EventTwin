# -*- coding: utf-8 -*-
"""查询层：双时间事实查询、混合检索（BM25+Dense RRF + Jev 相关性重排，v3.1）、带证据流式问答。

对齐 04 §12：
- 四类时间问答：当前状态 / 历史实际情况 / 当时系统知道什么 / 某段时间发生了什么；
- evidence pack：facts + events + evidence + corrections + conflicts + coverage；
- 无证据不用模型常识伪造当前状态（abstain），引用可回原文核验。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from . import config as cfg, llm, observe
from .nlp import BM25Index, rrf
from .pipeline.assertions import PRED_LABELS, scope_key as fmt_scope
from .store import vectors
from .util import CST, now_iso, parse_datetime, iso as iso_fmt

# ---------------------------------------------------------------------------
# 事实查询（双时间）
# ---------------------------------------------------------------------------

def current_fact(conn, slot_key: str, *, valid_as_of: str | None = None,
                 known_as_of: str | None = None) -> dict:
    """查询某槽位的选择历史（默认 now/now）。

    双时间过滤：valid_during ∋ valid_as_of 且 sys_during ∋ known_as_of。
    由 SQL 而非 LLM 实现"当时已知"与"现在回看"（04 §15.3）。
    """
    va = valid_as_of or now_iso()
    ka = known_as_of or now_iso()
    rows = conn.execute(
        "SELECT s.*, a.predicate, a.value_json, a.scope_json, a.subject_entity_id, "
        "a.correction_of, a.retraction_of, a.status AS assertion_status, a.document_version_id "
        "FROM slot_selection_history s LEFT JOIN assertion a "
        "ON a.assertion_id=s.chosen_assertion_id "
        "WHERE s.slot_key=? AND s.valid_from<=? AND (s.valid_to IS NULL OR s.valid_to>?) "
        "AND s.sys_from<=? AND (s.sys_to IS NULL OR s.sys_to>?) "
        "ORDER BY s.sys_from DESC LIMIT 5", (slot_key, va, va, ka, ka)).fetchall()
    ent = conn.execute(
        "SELECT e.canonical_name FROM entity e JOIN assertion a ON a.subject_entity_id=e.entity_id "
        "WHERE a.slot_key=? LIMIT 1", (slot_key,)).fetchone()
    out = {"slot_key": slot_key, "valid_as_of": va, "known_as_of": ka,
           "subject": ent["canonical_name"] if ent else None,
           "selections": [], "corrections": [], "conflicts": []}
    for r in rows:
        sel = dict(r)
        if sel["value_json"]:
            sel["value"] = json.loads(sel["value_json"])
        if sel["correction_of"]:
            old = conn.execute("SELECT value_json, created_at FROM assertion WHERE assertion_id=?",
                               (sel["correction_of"],)).fetchone()
            if old:
                out["corrections"].append({"old": json.loads(old["value_json"]),
                                           "old_assertion": sel["correction_of"],
                                           "corrected_at": sel["created_at"]})
        if sel["disposition"] == "conflicted":
            for cid in json.loads(sel["candidate_ids_json"] or "[]"):
                a = conn.execute("SELECT value_json, document_version_id, created_at FROM assertion "
                                 "WHERE assertion_id=?", (cid,)).fetchone()
                if a:
                    ev = conn.execute(
                        "SELECT quote, url FROM assertion_evidence WHERE assertion_id=? LIMIT 1",
                        (cid,)).fetchone()
                    out["conflicts"].append({
                        "assertion_id": cid, "value": json.loads(a["value_json"]),
                        "recorded_at": a["created_at"],
                        "evidence": {"quote": ev["quote"], "url": ev["url"]} if ev else None})
        out["selections"].append(sel)
    if not out["selections"]:
        out["empty_reason"] = "该时间口径下无选择记录（未知/未覆盖）"
    return out


def timeline(conn, *, entity_id: str | None = None, process_id: str | None = None,
             event_type: str | None = None, limit: int = 100) -> list[dict]:
    q = ("SELECT c.*, p.title AS process_title FROM event_cluster c "
         "LEFT JOIN process p ON p.process_id=c.process_id "
         "WHERE c.deleted_at IS NULL AND c.state NOT IN ('redirected','split','deleted') ")
    args: list = []
    if entity_id:
        q += ("AND c.cluster_id IN (SELECT to_id FROM semantic_relation "
              "WHERE relation='participates_in' AND from_type='entity' AND from_id=? "
              "AND deleted_at IS NULL) ")
        args.append(entity_id)
    if process_id:
        q += "AND c.process_id=? "
        args.append(process_id)
    if event_type:
        q += "AND c.event_type=? "
        args.append(event_type)
    q += "ORDER BY c.event_time_lower IS NULL, c.event_time_lower LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(q, args).fetchall():
        d = dict(r)
        d.pop("centroid", None)  # BLOB 不可 JSON 序列化
        d["members"] = conn.execute(
            "SELECT COUNT(*) n FROM cluster_membership WHERE cluster_id=? AND removed_at IS NULL",
            (r["cluster_id"],)).fetchone()["n"]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# 混合检索（BM25 + Dense + RRF + Rerank）
# ---------------------------------------------------------------------------

_doc_idx_cache: dict = {}


def _doc_index(conn) -> BM25Index:
    key = conn.execute("SELECT COUNT(*) n, COALESCE(MAX(rowid),0) m FROM document_version"
                       ).fetchone()
    ck = f"{key['n']}:{key['m']}"
    if _doc_idx_cache.get("key") == ck:
        return _doc_idx_cache["idx"]
    idx = BM25Index()
    for r in conn.execute(
            "SELECT dv.document_version_id, dv.title, dv.normalized_text, s.name "
            "FROM document_version dv JOIN document d ON d.document_id=dv.document_id "
            "LEFT JOIN source s ON s.source_id=d.source_id "
            "WHERE dv.status IN ('extracted','resolved','indexed','published') "
            "AND dv.normalized_text IS NOT NULL").fetchall():
        idx.add(r["document_version_id"], f"{r['title'] or ''} {r['name'] or ''} "
                                          f"{r['normalized_text'][:4000]}")
    _doc_idx_cache.update({"key": ck, "idx": idx})
    return idx


def search(conn, query: str, *, top_k: int | None = None, with_rerank: bool = True) -> dict:
    """BM25 + 向量 RRF 融合，再 Jev 重排。返回按事件去重后的证据列表。"""
    conf = cfg.load_config()
    k = top_k or conf["queries"]["search_top_k"]
    bm = _doc_index(conn).top(query, 30)
    qvec = llm.embed([query], stage="search.query")[0]
    dense = [oid for _t, oid, _s, _sc in vectors.cosine_search(conn, qvec, "doc_chunk", 30)]
    dense_frames = [oid for _t, oid, _s, _sc in vectors.cosine_search(conn, qvec, "mention_frame", 20)]
    fused = rrf([bm, dense, dense_frames])
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:30]
    # 取回原文片段：优先命中句
    items = []
    for docv_id, score in ranked:
        r = conn.execute(
            "SELECT dv.document_version_id, dv.title, dv.normalized_text, dv.published_at, "
            "d.canonical_url, s.name AS source_name FROM document_version dv "
            "JOIN document d ON d.document_id=dv.document_id "
            "LEFT JOIN source s ON s.source_id=d.source_id "
            "WHERE dv.document_version_id=?", (docv_id,)).fetchone()
        if r is None:
            continue
        snippet = _best_snippet(r["normalized_text"] or "", query)
        items.append({"document_version_id": docv_id, "title": r["title"],
                      "url": r["canonical_url"], "source": r["source_name"],
                      "published_at": r["published_at"], "snippet": snippet,
                      "rrf": round(score, 5)})
    if with_rerank and items:
        try:
            probs = llm.jev_choice_rank(
                query, [f"{i['title']} {i['snippet']}" for i in items],
                stage="search.rerank")
            for i, item in enumerate(items):
                item["relevance"] = round(probs[i], 4) if i < len(probs) else 0.0
            items.sort(key=lambda x: -x.get("relevance", 0))
        except llm.LLMError:
            pass
    return {"query": query, "channels": {"bm25": len(bm), "dense_doc": len(dense),
                                         "dense_mention": len(dense_frames)},
            "results": items[:k]}


def _best_snippet(text: str, query: str, width: int = 220) -> str:
    from .nlp import tokenize
    qtoks = set(tokenize(query)) - set(tokenize("的了在是和与及"))
    best_pos, best_hit = 0, 0
    for pos in range(0, max(len(text) - width, 0), 60):
        window = text[pos:pos + width]
        hits = sum(1 for t in qtoks if t in window)
        if hits > best_hit:
            best_hit, best_pos = hits, pos
    snip = text[best_pos:best_pos + width].replace("\n", " ")
    return (snip[:200] + "…") if len(snip) > 200 else snip


# ---------------------------------------------------------------------------
# 带证据问答（Evidence-backed QA，流式）
# ---------------------------------------------------------------------------

QA_SYSTEM = """你是企业外部情报系统的问答助手。严格依据【证据包】回答用户问题。

规则：
1. 只使用证据包中的事实与证据，不得引入外部常识或猜测；
2. 每个关键结论后标注引用编号，如 [1][2]；引用编号对应证据包中的 evidence 列表；
3. 证据包标记 conflict 的内容要说明"存在争议"并列出双方值；
4. 证据不足以回答时，明确说"根据现有证据无法回答"并说明缺什么，不要编造；
5. 涉及时间口径（当前/历史/当时已知）时，明确说出你依据的是哪个口径；
6. 用中文回答，条理清晰，先给结论再给依据。"""


# ---------------------------------------------------------------------------
# EventRAG 主检索路径（S09 事件网关 + Chronos Φent/Φtemp + Re³ 路由）
# ---------------------------------------------------------------------------

_EVENT_BM25: dict = {}


def _event_card_index(conn):
    """事件卡 BM25 索引（派生层，水位失效重建）。"""
    key = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(version),0) v FROM event_cluster "
        "WHERE deleted_at IS NULL AND state NOT IN ('redirected','split','deleted')"
    ).fetchone()
    ck = f"{key['n']}:{key['v']}"
    if _EVENT_BM25.get("key") == ck:
        return _EVENT_BM25["idx"]
    idx = BM25Index()
    for r in conn.execute(
            "SELECT cluster_id, card_text, summary FROM event_cluster "
            "WHERE deleted_at IS NULL AND state NOT IN ('redirected','split','deleted') "
            "AND card_text<>''").fetchall():
        idx.add(r["cluster_id"], r["card_text"] or r["summary"])
    _EVENT_BM25.update({"key": ck, "idx": idx})
    return idx


def _expand_entity_family(conn, entity_ids: list[str]) -> set[str]:
    """实体家族扩展：沿 seed 层级边（subsidiary_of/brand_of/related_to）扩一跳。

    这让"小米"能命中"小米汽车"的事件（用户指出的问题），同时保持边来源为
    权威词表（asserted_by='seed'），不是 LLM 语义边（S08 SimGraph 教训）。
    """
    fam = set(entity_ids)
    ph = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"SELECT from_id, to_id FROM semantic_relation WHERE relation IN "
        f"('subsidiary_of','brand_of','related_to','refers_to') AND deleted_at IS NULL "
        f"AND (from_id IN ({ph}) OR to_id IN ({ph}))", (*entity_ids, *entity_ids)).fetchall()
    for r in rows:
        fam.add(r["from_id"])
        fam.add(r["to_id"])
    return fam


def analyze_query(conn, query: str) -> dict:
    """查询分析：实体链接（含家族扩展）+ 时间窗（含缓冲带）+ 意图路由。

    规则优先（LongMemEval E.4：弱模型抽时间会幻觉剪枝，时间解析失败时不过滤）。
    """
    import re
    q = query or ""
    ents = _match_entities(conn, q)
    ent_ids = [e["entity_id"] for e in ents]
    family = _expand_entity_family(conn, ent_ids) if ent_ids else set()
    # 时间窗
    now = now_iso()
    win_from = win_to = None
    relative = None
    m = re.search(r"(近|过去)(一|两|三|1|2|3)年", q)
    if m:
        n = {"一": 1, "两": 2, "三": 3, "1": 1, "2": 2, "3": 3}[m.group(2)]
        relative = ("years", n)
    m2 = re.search(r"(近|过去)(一|两|三|六|1|2|3|6)个?月", q)
    if m2 and not relative:
        n = {"一": 1, "两": 2, "三": 3, "六": 6, "1": 1, "2": 2, "3": 3, "6": 6}[m2.group(2)]
        relative = ("months", n)
    if relative:
        from datetime import datetime
        base, _ = parse_datetime(now)
        if relative[0] == "years":
            win_from = iso_fmt(datetime(base.year - relative[1], base.month, base.day,
                                        tzinfo=CST))
        else:
            mm = base.month - relative[1]
            yy = base.year
            while mm <= 0:
                mm += 12
                yy -= 1
            win_from = iso_fmt(datetime(yy, mm, base.day, tzinfo=CST))
        win_to = now
    else:
        t = _find_date(q)
        if t:
            win_from = t
            win_to = now
    # 意图
    if re.search(r"当时(系统)?(知道|了解|认为)|当时已知", q):
        intent = "as_of"
    elif re.search(r"变化|变过|几次|历程|演化|历史|一路|以来|趋势|对比|比较|之前和现在|多少钱.*现在", q):
        intent = "evolution"
    elif re.search(r"当前|现在|目前|最新|如今的?|现状", q):
        intent = "current"
    else:
        intent = "exploration"
    return {"entities": ents, "entity_ids": ent_ids, "family_ids": family,
            "win_from": win_from, "win_to": win_to, "intent": intent}


def retrieve_events(conn, query: str, analysis: dict, *, top_k: int = 12) -> dict:
    """EventRAG 事件检索：事件卡 ANN + 实体结构化 + 事件卡 BM25 → RRF →
    时间软分（Chronos exp(-Δ/τ)，evolution 意图禁 recency）→ rerank 取 top-k。
    """
    fam = analysis["family_ids"]
    channels: dict[str, list[str]] = {}
    try:
        qvec = llm.embed([query], stage="search.event_query")[0]
        hits = vectors.cosine_search(conn, qvec, "event_card", 50)
        channels["ann_event_card"] = [oid for _t, oid, _s, _sc in hits]
    except Exception:  # noqa: BLE001
        channels["ann_event_card"] = []
    if fam:
        ph = ",".join("?" for _ in fam)
        rows = conn.execute(
            f"SELECT DISTINCT to_id FROM semantic_relation WHERE relation='participates_in' "
            f"AND from_type='entity' AND from_id IN ({ph}) AND deleted_at IS NULL LIMIT 800",
            tuple(fam)).fetchall()
        channels["entity_structured"] = [r["to_id"] for r in rows]
    else:
        channels["entity_structured"] = []
    try:
        channels["bm25_event_card"] = _event_card_index(conn).top(query, 30)
    except Exception:  # noqa: BLE001
        channels["bm25_event_card"] = []
    pool = []
    for ch in ("ann_event_card", "entity_structured", "bm25_event_card"):
        for cid in channels[ch]:
            if cid not in pool:
                pool.append(cid)
    if not pool:
        return {"events": [], "channels": {k: len(v) for k, v in channels.items()},
                "coverage_ok": False}
    # 载入事件行
    clusters = {}
    for cid in pool[:300]:
        r = conn.execute(
            "SELECT cluster_id, event_type, state, summary, card_text, event_time_lower, "
            "event_time_upper, process_id, version FROM event_cluster WHERE cluster_id=? "
            "AND deleted_at IS NULL AND state NOT IN ('redirected','split','deleted')",
            (cid,)).fetchone()
        if r is not None:
            clusters[cid] = dict(r)
    ranked_ids = [c for c in pool if c in clusters]
    # 时间软分（窗口外缓冲带内保留但降权；无窗口不过滤）
    tau_days = 180  # Chronos 附录 D 初值
    def _time_boost(c: dict) -> float:
        wf, wt = analysis["win_from"], analysis["win_to"]
        if not wf:
            return 1.0
        import math
        t_lo = parse_datetime(c["event_time_lower"])[0] if c["event_time_lower"] else None
        if t_lo is None:
            return 0.6  # 未知时间不硬排除（recall.py 同口径）
        if wf <= iso_fmt(t_lo) and (wt is None or iso_fmt(t_lo) <= wt):
            return 1.0
        delta = abs((parse_datetime(wf)[0] - t_lo).days)
        return math.exp(-delta / tau_days)
    # Jev 相关性分布（P_same 式，对事件卡；v3 替代 reranker）
    scored: list[tuple[str, float]] = []
    if ranked_ids:
        cards = [clusters[c]["card_text"] or clusters[c]["summary"] for c in ranked_ids[:60]]
        probs = []
        try:
            probs = llm.jev_choice_rank(query, cards, stage="search.event_rerank")
        except llm.LLMError:
            pass
        for i, cid in enumerate(ranked_ids[:60]):
            rel = (probs[i] if i < len(probs) else 0.5) * _time_boost(clusters[cid])
            # 结构化通道命中（同实体）加权：实体明确时优先实体自己的事件
            ent_bonus = 0.15 if cid in set(channels["entity_structured"]) and fam else 0.0
            scored.append((cid, rel + ent_bonus))
        scored.sort(key=lambda x: -x[1])
    events = []
    for cid, sc in scored[:top_k]:
        c = clusters[cid]
        c["score"] = round(sc, 4)
        events.append(c)
    return {"events": events, "channels": {k: len(v) for k, v in channels.items()},
            "coverage_ok": bool(scored)}


def expand_event_gateway(conn, events: list[dict], analysis: dict) -> dict:
    """事件网关一跳扩展（E²RAG/S08：1-hop 最优）：
    命中事件 → 成员证据 + 断言值/历史 + 邻接事件（同过程 precedes ±1、同实体时间邻域）。
    """
    ev_ids = {e["cluster_id"] for e in events}
    evidence, context_events = [], []
    seen_docv = set()
    for e in events:
        cid = e["cluster_id"]
        members = conn.execute(
            "SELECT m.mention_id, m.event_type, m.action, m.evidence_json, m.claims_json, "
            "m.event_time_lower, dv.document_version_id, dv.title, dv.published_at, "
            "d.canonical_url, s.name AS source_name, sl.lineage_group "
            "FROM cluster_membership cm JOIN event_mention m ON m.mention_id=cm.mention_id "
            "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
            "LEFT JOIN document d ON d.document_id=dv.document_id "
            "LEFT JOIN source s ON s.source_id=d.source_id "
            "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
            "WHERE cm.cluster_id=? AND cm.removed_at IS NULL AND m.status='valid' "
            "ORDER BY COALESCE(dv.published_at, cm.added_at) LIMIT 8", (cid,)).fetchall()
        for mrow in members:
            if mrow["document_version_id"] in seen_docv and mrow["lineage_group"]:
                continue
            seen_docv.add(mrow["document_version_id"])
            for ev in json.loads(mrow["evidence_json"] or "[]")[:1]:
                evidence.append({
                    "cluster_id": cid, "quote": ev.get("quote", ""),
                    "title": mrow["title"], "url": mrow["canonical_url"],
                    "source": mrow["source_name"], "published_at": mrow["published_at"],
                    "action": mrow["action"]})
        # 断言（当前值 + 历史版本）
        for a in conn.execute(
                "SELECT assertion_id, predicate, value_json, valid_from, valid_to, status "
                "FROM assertion WHERE event_id=? ORDER BY valid_from LIMIT 6", (cid,)).fetchall():
            e.setdefault("assertions", []).append(dict(a))
        # 邻接：同过程事件
        if e["process_id"]:
            for nb in conn.execute(
                    "SELECT cluster_id, summary, event_type, event_time_lower, card_text "
                    "FROM event_cluster WHERE process_id=? AND cluster_id<>? "
                    "AND deleted_at IS NULL AND state NOT IN ('redirected','split','deleted') "
                    "ORDER BY event_time_lower LIMIT 6", (e["process_id"], cid)).fetchall():
                if nb["cluster_id"] not in ev_ids and                         nb["cluster_id"] not in {c["cluster_id"] for c in context_events}:
                    context_events.append(dict(nb))
    return {"evidence": evidence[:24], "context_events": context_events[:8]}


def _slot_history(conn, entity_ids, *, valid_as_of=None, known_as_of=None) -> list[dict]:
    """实体槽位演化史：当前值 + 全部版本区间（Chronos Φent 的数据源）。"""
    out = []
    for eid in entity_ids:
        for r in conn.execute(
                "SELECT DISTINCT slot_key FROM assertion WHERE subject_entity_id=? LIMIT 40",
                (eid,)).fetchall():
            f = current_fact(conn, r["slot_key"], valid_as_of=valid_as_of,
                             known_as_of=known_as_of)
            if not f["selections"]:
                continue
            # 全版本序列（同一槽位的全部选择行按有效时间排）
            versions = []
            for v in conn.execute(
                    "SELECT s.valid_from, s.valid_to, s.disposition, a.value_json "
                    "FROM slot_selection_history s LEFT JOIN assertion a "
                    "ON a.assertion_id=s.chosen_assertion_id WHERE s.slot_key=? "
                    "ORDER BY s.valid_from, s.sys_from", (r["slot_key"],)).fetchall():
                versions.append({"valid_from": (v["valid_from"] or "")[:10],
                                 "valid_to": (v["valid_to"] or "开放")[:10],
                                 "value": json.loads(v["value_json"]).get("value")
                                 if v["value_json"] else None,
                                 "disposition": v["disposition"]})
            f["versions"] = versions
            out.append(f)
    return out


def build_evidence_pack(conn, query: str) -> dict:
    """EventRAG 证据包：事件网关 + Φent 实体演化链 + Φtemp 全局时间线 + 结构化事实。"""
    analysis = analyze_query(conn, query)
    intent = analysis["intent"]
    va = ka = now_iso()
    if intent == "as_of":
        t = _find_date(query)
        va = ka = t or now_iso()
    ret = retrieve_events(conn, query, analysis)
    gw = expand_event_gateway(conn, ret["events"], analysis) if ret["events"] else \
        {"evidence": [], "context_events": []}
    # 结构化事实与演化史
    facts, chains = [], []
    if analysis["entity_ids"]:
        facts = _slot_history(conn, analysis["entity_ids"],
                              valid_as_of=None if intent == "evolution" else va,
                              known_as_of=None if intent == "evolution" else ka)
        # Φent：每实体事件链（时间排序）
        for e in analysis["entities"]:
            evs = timeline(conn, entity_id=e["entity_id"], limit=40)
            if evs:
                chains.append({"entity": e["canonical_name"], "events": [
                    {"date": (x["event_time_lower"] or "?")[:10],
                     "type": x["event_type"], "summary": (x["card_text"] or x["summary"] or "")[:90]}
                    for x in evs]})
    all_events = ret["events"] + gw["context_events"]
    # Φtemp：全局时间线（事件按时间排序）
    tline = sorted(all_events, key=lambda x: x.get("event_time_lower") or "9999")
    evidence = []
    for i, ev in enumerate(gw["evidence"], 1):
        evidence.append({"no": i, "title": ev["title"], "url": ev["url"], "source": ev["source"],
                         "snippet": ev["quote"], "published_at": ev["published_at"]})
    if not evidence:  # 事件通道空 → chunk RAG 兜底
        s = search(conn, query, with_rerank=True)
        for i, item in enumerate(s["results"][:6], 1):
            evidence.append({"no": i, "title": item["title"], "url": item["url"],
                             "source": item["source"], "snippet": item["snippet"],
                             "published_at": item["published_at"]})
    pack = {
        "query": query, "query_mode": intent,
        "valid_as_of": va, "known_as_of": ka,
        "analysis": {"entities": [e["canonical_name"] for e in analysis["entities"]],
                     "family_size": len(analysis["family_ids"]),
                     "win_from": analysis["win_from"], "win_to": analysis["win_to"]},
        "retrieval": ret["channels"],
        "events": [{"cluster_id": e["cluster_id"], "type": e["event_type"],
                    "card": (e.get("card_text") or e["summary"] or "")[:140],
                    "time_lower": e["event_time_lower"], "score": e.get("score"),
                    "assertions": e.get("assertions", [])} for e in ret["events"]],
        "timeline": [{"date": (e.get("event_time_lower") or "?")[:10],
                      "card": (e.get("card_text") or e.get("summary") or "")[:120]}
                     for e in tline[:30]],
        "chains": chains[:6],
        "facts": facts[:8],
        "evidence": evidence,
        "conflicts": [c for f in facts for c in f.get("conflicts", [])][:6],
        "corrections": [c for f in facts for c in f.get("corrections", [])][:6],
        "coverage": {"events": len(ret["events"]), "context": len(gw["context_events"]),
                     "evidence": len(evidence),
                     "retrieval_complete": bool(ret["coverage_ok"] or evidence)},
    }
    return pack


def _slim_pack(pack: dict) -> dict:
    """给前端的证据包瘦身。"""
    slim = dict(pack)
    slim["facts"] = [{"slot_key": f["slot_key"], "subject": f["subject"],
                      "current": [s.get("value", {}).get("value") if s.get("value") else None
                                  for s in f["selections"][:1]],
                      "versions": f.get("versions", [])[:12],
                      "disposition": [s["disposition"] for s in f["selections"][:1]],
                      "valid_as_of": f["valid_as_of"], "known_as_of": f["known_as_of"]}
                     for f in pack["facts"]]
    return slim


def answer_stream(conn, query: str, *, use_cache: bool = False):
    """生成器：先 yield 证据包事件，再逐段 yield 回答 token，最后 yield 完成标记。"""
    pack = build_evidence_pack(conn, query)
    observe.emit(conn, "qa", f"问答: {query[:60]}", kind="qa.start",
                 data={"mode": pack["query_mode"], "events": pack["coverage"]["events"],
                       "evidence": pack["coverage"]["evidence"]})
    yield {"type": "evidence", "pack": _slim_pack(pack)}
    user = _qa_user_prompt(pack, query)
    import requests as _rq
    c = cfg.load_config()["models"]
    msgs = [{"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": user}]
    full = []
    try:
        resp = _rq.post(c["chat_base_url"] + "/chat/completions", json={
            "model": c["chat_model"], "messages": msgs, "temperature": 0.3,
            "max_tokens": cfg.load_config()["queries"]["answer_max_tokens"], "stream": True,
            # vLLM 的 Qwen3 不识别 /no_think 软开关，思考关闭走模板参数（与 llm.chat 同口径）
            "chat_template_kwargs": {"enable_thinking": False},
        }, stream=True, timeout=c["chat_timeout_seconds"],
            headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                chunk = obj["choices"][0]
            except (KeyError, IndexError):
                continue
            piece = chunk.get("delta") or chunk.get("message") or {}
            delta = piece.get("content")
            if delta:
                full.append(delta)
                yield {"type": "token", "text": delta}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"生成失败: {e}"}
        return
    yield {"type": "done", "answer": "".join(full), "citations": pack["evidence"]}
    observe.emit(conn, "qa", f"问答完成: {query[:40]} → {len(''.join(full))} 字",
                 kind="qa.done")


def _qa_user_prompt(pack: dict, query: str) -> str:
    facts = []
    for f in pack["facts"]:
        cur = f["selections"][0] if f["selections"] else None
        if cur:
            facts.append(f"- {f['subject']}｜{f['slot_key'].split(':', 1)[-1]}："
                         f"当前值 {json.dumps(cur.get('value') or {}, ensure_ascii=False)}"
                         f"（{cur['disposition']}）")
        for v in f.get("versions", [])[:10]:
            facts.append(f"  · {v['valid_from']}~{v['valid_to']}：{v['value']}"
                         f"（{v['disposition']}）")
    tline = [f"- {t['date']}：{t['card']}" for t in pack.get("timeline", [])[:24]]
    chains = []
    for c in pack.get("chains", []):
        chains.append(f"◆ {c['entity']} 演化链：" + " → ".join(
            f"{e['date']}{e['summary'][:40]}" for e in c["events"][:12]))
    evs = [f"- [{e.get('no')}] 事件：{e['card']}" for e in []]
    quotes = [f"- [{e['no']}] {e['title']}（{e['source']}，{(e['published_at'] or '')[:10]}）："
              f"{e['snippet']}" for e in pack["evidence"]]
    parts = [
        f"【查询口径】intent={pack['query_mode']}，valid_as_of={pack['valid_as_of'][:19]}，"
        f"known_as_of={pack['known_as_of'][:19]}",
        f"【命中的查询实体】{'、'.join(pack['analysis']['entities']) or '无'}",
        "【结构化事实（SQL 双时间账本，非模型推断）】", *(facts[:28] or ["- 无"]),
        "【实体事件演化链（按发生时间）】", *(chains or ["- 无"]),
        "【窗口内时间线（Φtemp）】", *(tline or ["- 无"]),
        "【原文证据（引用编号来源）】", *(quotes or ["- 无"]),
        f"【冲突】{json.dumps(pack['conflicts'], ensure_ascii=False)[:500] if pack['conflicts'] else '无'}",
        f"【更正】{json.dumps(pack['corrections'], ensure_ascii=False)[:500] if pack['corrections'] else '无'}",
        f"【覆盖】{json.dumps(pack['coverage'], ensure_ascii=False)}",
        f"\n【用户问题】{query}\n"
        "回答要求：先给结论再给依据；引用用 [编号]；时间线类问题按时间顺序列点；"
        "表述简洁直接，不堆砌背景；证据不足处明确说明，不编造。",
    ]
    return "\n".join(parts)


def _route_query(query: str) -> tuple[str, str, str]:
    """查询路由：规则优先解析时间口径（04 §12.1）。"""
    now = now_iso()
    q = query or ""
    # 当时系统知道什么（known_as_of 固定到过去）
    import re
    m = re.search(r"(当时|那时|之前)\s*系统?\s*(知道|了解|认为)", q) or \
        re.search(r"(当时已知|系统在.{0,12}知道)", q)
    if m:
        t = _find_date(q) or now
        return "known_as_of", t, t if t != now else now  # valid=known=过去时点
    # 历史实际情况（现在回看）
    if re.search(r"(回看|实际上|实际是|后来|更正后)", q):
        t = _find_date(q) or now
        return "historical_valid", t, now
    # 截至/在某某时
    t = _find_date(q)
    if t:
        return "valid_as_of", t, now
    return "current", now, now


def _find_date(q: str) -> str | None:
    import re
    m = re.search(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})[日号]?", q)
    if m:
        try:
            return iso_fmt(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 12,
                                    tzinfo=CST))
        except ValueError:
            return None
    m = re.search(r"(\d{4})[-年](\d{1,2})月?", q)
    if m:
        try:
            return iso_fmt(datetime(int(m.group(1)), int(m.group(2)), 15, 12, tzinfo=CST))
        except ValueError:
            return None
    return None


def _match_entities(conn, query: str) -> list[dict]:
    """规则实体匹配：规范名/别名出现在查询中。"""
    from .pipeline.entities import norm_key
    hits = []
    for r in conn.execute("SELECT entity_id, canonical_name, norm_key FROM entity "
                          "WHERE deleted_at IS NULL").fetchall():
        if r["norm_key"] and r["norm_key"] in norm_key(query):
            hits.append(dict(r))
            continue
        for a in conn.execute("SELECT norm_key FROM entity_alias WHERE entity_id=?",
                              (r["entity_id"],)).fetchall():
            if a["norm_key"] and len(a["norm_key"]) >= 2 and a["norm_key"] in norm_key(query):
                hits.append(dict(r))
                break
    return hits[:5]
