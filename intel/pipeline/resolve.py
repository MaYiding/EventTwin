# -*- coding: utf-8 -*-
"""同事件判别与在线事件归并（04 §8；v3 双模型版）——本系统的核心决策模块。

评分骨架 S(x,C) = w1·S_semantic + w2·P_same + w3·S_time + w4·S_entity
                + w5·S_location + w6·S_type（排序分数，不天然等于概率）。
决策四分支：attach / create_provisional / judge（灰区 Jev）/ pending。

v3.1 三件套（qwen3.8-27b 生成 + qwen3-embedding-8b 召回 + Jev 判定；依据
2026-09-24 五模型评测：Jev 事件核对 1.0 / 实体核对 0.966；生成模型判定面
"宽容判官"、embedding 判定面"全拒型"）：
- P_same 特征与灰区判别统一由 Jev 承担：
  · 排序阶段：一次 choice 对 top 候选给出同事件概率分布（_jev_psame）；
  · 灰区判别：choice（同事件/新事件/相关过程）+ noul（证据充分性）双问
    （_llm_judge），身份锚点/冲突由规则推导留痕；
- S_semantic 特征用回质心余弦（embedding-8b 回归，只做召回与特征、不做判定）；
  BM25 词面分保留为无质心时的兜底（首个成员/向量缺口时）；
- 事件卡文本 + 向量恢复（EventRAG 检索单元）。

铁律：
- 分数只用于排序与低分新建判据，不用于未校准自动归并；
- 高置信直通带必须三重结构锚点（同类型 + 共享实体 + 严格时间重叠）；
- attach 只影响"新提及进旧簇"；簇间合并 merge_clusters 属高影响操作，仅人工（graph_service）。
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from .. import config as cfg, llm, observe
from ..store import vectors
from ..util import det_uuid, now_iso, sha256_text
from . import recall as recall_mod

POLICY_VERSION = "resolve-policy-v3.1-jev-emb8b"

# 待批量嵌入事件卡的簇（片尾/阶段末统一处理）
_PENDING_CARD_EMBEDS: set = set()


def flush_card_embeds(conn: sqlite3.Connection) -> int:
    """批量补齐事件卡向量（16/批；命中向量缓存时近零成本）。"""
    if not _PENDING_CARD_EMBEDS:
        return 0
    from ..store import vectors as V  # 函数内别名（模块级导入名为 vectors）
    conf_cfg = cfg.load_config()
    model_id = conf_cfg["models"]["embed_model"]
    dim = conf_cfg["models"]["embed_dimensions"]
    ids = sorted(_PENDING_CARD_EMBEDS)  # 确定性顺序
    _PENDING_CARD_EMBEDS.clear()
    n = 0
    B = 16
    for i in range(0, len(ids), B):
        batch = ids[i:i + B]
        rows = [dict(r) for r in conn.execute(
            f"SELECT cluster_id, card_text, card_hash FROM event_cluster "
            f"WHERE cluster_id IN ({','.join('?' * len(batch))})", tuple(batch)).fetchall()]
        todo = [r for r in rows if r["card_text"] and r["card_hash"]]
        if not todo:
            continue
        try:
            mat = llm.embed([r["card_text"] for r in todo], stage="embed.event_card")
            for r, vec in zip(todo, mat):
                V.add_vector(conn, role="event_card", owner_type="cluster",
                             owner_id=r["cluster_id"], sub_id="", model_id=model_id,
                             dim=dim, vec=vec, text_hash=r["card_hash"])
                n += 1
            conn.commit()
        except Exception as _e:  # noqa: BLE001
            observe.emit(conn, "vector", f"事件卡批量嵌入失败: {_e}", level="warn",
                         kind="vector.card_fail")
    return n

# 同事件判定规则（进入 Jev state；与 v2 JUDGE_SYSTEM 的标准一致，逐条对应）
JEV_RULES = """同事件判定规则：
- 原子事件 = 特定参与方 + 特定对象 + 特定时间 + 一次具体动作/阶段转换；
- 同一次动作被不同媒体报道（含转载、中英文）= 同一事件；
- 同一产品先后两次不同调价/两次不同发布 = 不同事件（可属同一过程）；
- 并购的宣布与交割 = 不同事件；专利在不同国家公开 = 不同事件；
- 旧闻被重新报道 = 同一事件（按事件发生时间判断，不是报道时间）；
- 官方更正此前报道的金额/日期 = 同一事件（事实层另行处理）；
- 主题相近但动作、对象或时间不同 = 不同事件。"""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def resolve_mention(conn: sqlite3.Connection, mention_id: str, *, use_cache: bool = True) -> dict:
    plan = resolve_plan(conn, mention_id, use_cache=use_cache)
    return apply_plan(conn, plan)


def resolve_plan(conn: sqlite3.Connection, mention_id: str, *, use_cache: bool = True,
                 allowed_cluster_ids: set | None = None) -> dict:
    """计划阶段（只读 + 模型调用，可并行）：召回受限定的快照候选集，
    返回可执行的决策计划 dict；不写库。"""
    mention = conn.execute("SELECT * FROM event_mention WHERE mention_id=?", (mention_id,)).fetchone()
    if mention is None or mention["status"] != "valid":
        return {"kind": "skipped"}
    conf = cfg.load_config()
    observe.CURRENT_TRACE = mention_id

    # 已有活跃归属 → 幂等跳过
    active = conn.execute(
        "SELECT cluster_id FROM cluster_membership WHERE mention_id=? AND removed_at IS NULL",
        (mention_id,)).fetchone()
    if active:
        return {"skipped": True, "cluster": active["cluster_id"]}

    frame_vec = vectors.get_vector(conn, owner_type="mention", owner_id=mention_id,
                                   role="mention_frame")

    n_clusters = conn.execute(
        "SELECT COUNT(*) n FROM event_cluster WHERE deleted_at IS NULL "
        "AND state NOT IN ('redirected','split','deleted')").fetchone()["n"]

    # 1) 多路召回（限定时间片快照候选；ANN + BM25 + 结构化 + 未投影补丁）
    rec = recall_mod.recall_candidates(conn, mention, frame_vec,
                                       allowed_ids=allowed_cluster_ids)

    # 2) 空库首报：直接新建（召回健康按定义成立）
    if n_clusters == 0:
        return {"kind": "create", "mention_id": mention_id, "reason": "empty_store",
                "scores": {}, "action": "create_provisional", "judge": None, "rec": rec}

    # 3) 特征打分与排序（分数用于排序与低分判据，不用于未校准自动归并）
    ranked = _rank_candidates(conn, mention, rec["candidates"], frame_vec=frame_vec)
    anchor = _unique_id_anchor(conn, mention, rec["candidates"])

    cluster_conf = conf["cluster"]
    coverage_ok = rec["coverage_ok"]

    # 4) 决策
    if anchor:
        return {"kind": "attach", "mention_id": mention_id, "target": anchor, "ranked": ranked,
                "rec": rec, "action": "attach", "reason": "unique_id_anchor", "judge": None}
    if not ranked:
        if coverage_ok:
            return {"kind": "create", "mention_id": mention_id,
                    "reason": "no_candidates_after_filter", "scores": {},
                    "action": "create_provisional", "judge": None, "rec": rec}
        return {"kind": "pending", "mention_id": mention_id, "ranked": ranked, "rec": rec,
                "reason": "coverage_not_ok", "judge": None}
    best_score = ranked[0]["score"]
    # 4.5) 高置信直通带（TingIS s*>0.95 bypass 的结构化版）：
    #     分数达带 + 结构身份一致（同类型 + 共享实体 + 严格时间重叠）→ 免判别器归并。
    #     不是纯分数归并：三重结构锚点等价于复合唯一键，不违反 calibrated=false 铁律。
    if cluster_conf.get("high_confidence_bypass") and best_score >= \
            cluster_conf.get("bypass_score", 0.78):
        top = ranked[0]
        f = top["features"]
        if f.get("type") == 1.0 and f.get("entity", 0) >= 0.5 and f.get("time_known") \
                and f.get("time", 0) >= 0.999:
            return {"kind": "attach", "mention_id": mention_id, "target": top["cluster_id"],
                    "ranked": ranked, "rec": rec, "action": "attach",
                    "reason": "high_confidence_bypass", "judge": None}
    if best_score < cluster_conf["poc_score_new_floor"] and coverage_ok:
        return {"kind": "create", "mention_id": mention_id,
                "reason": f"low_score:{best_score:.2f}",
                "scores": {c["cluster_id"]: c["score"] for c in ranked},
                "action": "create_provisional", "judge": None, "rec": rec}
    if not coverage_ok:
        return {"kind": "pending", "mention_id": mention_id, "ranked": ranked, "rec": rec,
                "reason": "coverage_not_ok", "judge": None}

    # 5) 灰区：Top-3 提交 Jev 判别（必要时扩池复判一次）
    judged = _llm_judge(conn, mention, ranked, use_cache=use_cache)
    if judged is None or judged.get("decision") in ("insufficient",):
        return {"kind": "pending", "mention_id": mention_id, "ranked": ranked, "rec": rec,
                "reason": "judge_insufficient", "judge": judged}
    if judged.get("decision") == "request_more_candidates" and not rec.get("expanded"):
        rec2 = recall_mod.recall_candidates(conn, mention, frame_vec, expand=True,
                                            allowed_ids=allowed_cluster_ids)
        rec2["expanded"] = True
        ranked2 = _rank_candidates(conn, mention, rec2["candidates"], frame_vec=frame_vec)
        judged = _llm_judge(conn, mention, ranked2, use_cache=use_cache)
        if judged is None or judged.get("decision") in ("insufficient", "request_more_candidates"):
            return {"kind": "pending", "mention_id": mention_id, "ranked": ranked2, "rec": rec2,
                    "reason": "judge_still_unclear", "judge": judged}
        ranked, rec = ranked2, rec2
    dec = judged.get("decision")
    if dec == "same_event":
        target = _resolve_target(judged, ranked)
        if target:
            return {"kind": "attach", "mention_id": mention_id, "target": target,
                    "ranked": ranked, "rec": rec, "action": "judge_attach",
                    "reason": judged.get("reason_code") or "judge_same_event", "judge": judged}
        return {"kind": "pending", "mention_id": mention_id, "ranked": ranked, "rec": rec,
                "reason": "judge_target_invalid", "judge": judged}
    if dec in ("new_event", "related_process"):
        plan = {"kind": "create", "mention_id": mention_id,
                "reason": judged.get("reason_code") or f"judge_{dec}",
                "scores": {c["cluster_id"]: c["score"] for c in ranked},
                "action": "judge_create", "judge": judged, "rec": rec,
                "related_process_target": _resolve_target(judged, ranked)
                if dec == "related_process" else None}
        return plan
    return {"kind": "pending", "mention_id": mention_id, "ranked": ranked, "rec": rec,
            "reason": f"judge_unknown:{dec}", "judge": judged}


def apply_plan(conn: sqlite3.Connection, plan: dict, *, use_cache: bool = True) -> dict:
    """执行阶段（主线程串行）：把计划落库并触发派生更新。"""
    kind = plan.get("kind")
    mention = conn.execute("SELECT * FROM event_mention WHERE mention_id=?",
                           (plan.get("mention_id", ""),)).fetchone()
    if mention is None or kind == "skipped":
        return {"skipped": True}
    ranked = plan.get("ranked", [])
    rec = plan.get("rec") or {"coverage_ok": True, "channels": {}, "excluded": []}
    emit_recall_event(conn, mention, rec)
    frame_vec = vectors.get_vector(conn, owner_type="mention", owner_id=mention["mention_id"],
                                   role="mention_frame")
    if kind == "attach":
        return _finish_attach(conn, mention, plan["target"], ranked, rec,
                              action=plan.get("action", "attach"), reason=plan.get("reason", ""),
                              judge=plan.get("judge"), use_cache=use_cache)
    if kind == "create":
        out = _finish_create(conn, mention, frame_vec, reason=plan.get("reason", ""),
                             scores=plan.get("scores", {}), action=plan.get(
                                 "action", "create_provisional"),
                             judge=plan.get("judge"), use_cache=use_cache)
        tgt = plan.get("related_process_target")
        if tgt:
            _link_related_process(conn, out["cluster"], tgt)
        return out
    return _finish_pending(conn, mention, ranked, rec, reason=plan.get("reason", ""),
                           judge=plan.get("judge"))


# ---------------------------------------------------------------------------
# 特征与排序
# ---------------------------------------------------------------------------

def _rank_candidates(conn, mention, candidate_ids: list[str], *, frame_vec=None) -> list[dict]:
    if not candidate_ids:
        return []
    weights = cfg.load_config()["cluster"]["score_weights"]
    gap_conf = cfg.load_config()["cluster"]["hard_time_gap_days"]
    window_days = gap_conf.get(mention["event_type"] or "default",
                               gap_conf.get("default", 120))
    m_ents = set(recall_mod._mention_entity_ids(mention))
    m_scope = json.loads(mention["scope_json"] or "{}")

    clusters = []
    for cid in candidate_ids:
        c = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cid,)).fetchone()
        if c is None or c["deleted_at"] or c["state"] in ("redirected", "split", "deleted"):
            continue
        clusters.append(c)
    if not clusters:
        return []

    # S_semantic 特征：质心余弦（embedding-8b）；无质心候选（首成员/向量缺口）
    # 或无 frame 向量时兜底 BM25 词面分池内归一（确定性降级，不阻断）
    lex = recall_mod.cluster_lexical_scores(conn, mention["frame_text"])
    lex_top = max(lex.values(), default=0.0)

    # P_same 特征：Jev 同事件概率分布（判定权只在 Jev）
    p_same_map = _jev_psame(conn, mention, clusters, lex, m_ents)

    out = []
    for c in clusters:
        if frame_vec is not None:
            centroid = vectors.get_vector(conn, owner_type="cluster", owner_id=c["cluster_id"],
                                          role="cluster_centroid")
        else:
            centroid = None
        if centroid is not None and frame_vec is not None:
            s_sem = float(np.dot(frame_vec, centroid))
        else:
            s_sem = (lex.get(c["cluster_id"], 0.0) / lex_top) if lex_top > 0 else 0.0
        p_same = p_same_map.get(c["cluster_id"], 0.0)
        # 时间特征：重叠=1，按间隔线性衰减，未知=0.5（显式 mask，不当不一致）
        s_time, time_mask = _time_feature(mention, c, window_days)
        c_ents = _cluster_entity_ids(conn, c["cluster_id"])
        s_ent = _jaccard(m_ents, c_ents)
        s_loc = _location_feature(m_scope, (json.loads(c["frame_json"]) or {}).get("scope") or {})
        s_type = 1.0 if c["event_type"] == mention["event_type"] else 0.25
        score = (weights["semantic"] * s_sem + weights["same_event"] * p_same +
                 weights["time"] * (s_time * time_mask + 0.5 * (1 - time_mask)) +
                 weights["entity"] * (s_ent if c_ents else 0.5) +
                 weights["location"] * s_loc + weights["type"] * s_type)
        out.append({"cluster_id": c["cluster_id"], "score": round(float(score), 4),
                    "features": {"semantic": round(s_sem, 4), "same_event": round(p_same, 4),
                                 "time": round(s_time, 3), "time_known": bool(time_mask),
                                 "entity": round(s_ent, 3), "location": round(s_loc, 3),
                                 "type": s_type},
                    "cluster": c})
    # 同分候选按 cluster_id 决定性 tiebreak：无 tiebreak 时顺序依赖召回通道行序，
    # 会造成 decision 留痕漂移（业务结果不变，但重放指纹不一致）
    out.sort(key=lambda x: (-x["score"], x["cluster_id"]))
    return out


def _jev_psame(conn, mention, clusters: list, lex: dict[str, float],
               m_ents: set) -> dict[str, float]:
    """Jev 同事件概率分布：一次 choice 替代 reranker 批量相关性分。

    池先按（结构化先验, BM25 词面分, cluster_id）确定性预排序截断到
    jev_max_candidates（choice 选项上限 255，过长的选项列表也会稀释判断质量），
    未入选候选记 0。Jev 不可用（LLMError）时返回空 dict → P_same 全 0，
    与原 reranker 失败时取 0.5 中性值的差别：宁低勿高，触发低分新建而非误并。
    """
    conf = cfg.load_config()
    cap = int(conf["models"].get("jev_max_candidates", 24))
    if not clusters:
        return {}
    # 结构化先验：与新提及共享实体的历史簇排前
    ent_hit: set = set()
    if m_ents:
        ph = ",".join("?" * len(m_ents))
        rows = conn.execute(
            f"SELECT DISTINCT to_id FROM semantic_relation WHERE relation='participates_in' "
            f"AND from_type='entity' AND from_id IN ({ph}) AND deleted_at IS NULL",
            tuple(m_ents)).fetchall()
        ent_hit = {r["to_id"] for r in rows}
    order = sorted(clusters, key=lambda c: (0 if c["cluster_id"] in ent_hit else 1,
                                            -lex.get(c["cluster_id"], 0.0),
                                            c["cluster_id"]))[:cap]
    docs = []
    for c in order:
        reps = _representatives(c)
        t = f"{c['event_time_lower'] or '?'}~{c['event_time_upper'] or '?'}"
        rep = next((r.get("quote", "")[:50] for r in reps if r.get("quote")), "")
        docs.append(f"{c['summary']}｜类型{c['event_type']}｜时间{t}"
                    + (f"｜证据“{rep}”" if rep else ""))
    try:
        probs = llm.jev_choice_rank(
            mention["frame_text"], docs,
            instructions=(
                "判断 `查询`（新提及的事件框架）与 `候选` 中哪一项描述同一个现实发生的"
                "原子事件。同一事件 = 相同参与方/对象 + 相同时间 + 同一次动作；仅主题"
                "相近、或同一产品的另一次调价/发布、或并购宣布与交割，都不是同一事件。"),
            none_desc="新提及与所有候选都不是同一事件（应为新事件）",
            stage="jev.p_same")
    except llm.LLMError as e:
        observe.emit(conn, "resolve", f"Jev P_same 调用失败: {e}", level="warn",
                     kind="resolve.psame_error", target_id=mention["mention_id"])
        return {}
    return {c["cluster_id"]: p for c, p in zip(order, probs)}


def _time_feature(mention, cluster, window_days: int) -> tuple[float, float]:
    m_lo, m_hi = recall_mod._dt(mention["event_time_lower"]), recall_mod._dt(mention["event_time_upper"])
    c_lo, c_hi = recall_mod._dt(cluster["event_time_lower"]), recall_mod._dt(cluster["event_time_upper"])
    if m_lo is None or c_lo is None:
        return 0.5, 0.0  # 未知 → mask=0，取中性分
    m_hi = m_hi or m_lo
    c_hi = c_hi or c_lo
    if m_lo <= c_hi and c_lo <= m_hi:
        return 1.0, 1.0
    gap = (m_lo - c_hi).days if m_lo > c_hi else (c_lo - m_hi).days
    return max(0.0, 1.0 - gap / max(window_days, 1)), 1.0


def _location_feature(m_scope: dict, c_scope: dict) -> float:
    keys = ("market", "channel")
    hit, known = 0, 0
    for k in keys:
        mv, cv = m_scope.get(k), c_scope.get(k)
        if mv and cv:
            known += 1
            if str(mv).strip().upper() == str(cv).strip().upper():
                hit += 1
    if known == 0:
        return 0.5
    return hit / known


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _cluster_entity_ids(conn, cluster_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT from_id FROM semantic_relation WHERE relation='participates_in' "
        "AND to_type='event' AND to_id=? AND deleted_at IS NULL", (cluster_id,)).fetchall()
    return {r["from_id"] for r in rows}


def _unique_id_anchor(conn, mention, candidate_ids: list[str]) -> str | None:
    mb = (json.loads(mention["scope_json"] or "{}") or {}).get("business_no")
    if not mb:
        return None
    for cid in candidate_ids:
        c = conn.execute("SELECT frame_json FROM event_cluster WHERE cluster_id=?",
                         (cid,)).fetchone()
        if c and (json.loads(c["frame_json"]) or {}).get("scope", {}).get("business_no") \
                and str((json.loads(c["frame_json"]))["scope"]["business_no"]).strip() == str(mb).strip():
            return cid
    return None


def _representatives(cluster_row) -> list[dict]:
    try:
        return json.loads(cluster_row["frame_json"] or "{}").get("representatives") or []
    except json.JSONDecodeError:
        return []


from ..event_types import TYPE_LABELS  # 统一类型标签（16 类）


def build_event_card(members, conn) -> str:
    """生成事件卡文本（EventRAG 检索单元）：规则模板、确定性、可重建。

    对齐 S09 双流之"事件流"+ LongMemEval K=V+fact（索引期键扩展）：
    把类型/主体/对象/动作/时间/关键事实值/来源数都拼进被索引文本
    （v3 被索引渠道 = 事件卡 BM25 + Jev 相关性，卡片向量已随 embedding 移除）。
    不用 LLM 重写摘要——省调用且重放可复现（04 §11.3）。
    """
    latest = members[-1]
    actors = sorted({a.get("name", "") for m in members
                     for a in json.loads(m["actor_json"] or "[]") if isinstance(a, dict)})
    objects = sorted({o.get("name", "") for m in members
                      for o in json.loads(m["object_json"] or "[]") if isinstance(o, dict)})
    # 关键事实值：取成员 claims 中出现最多的谓词值对（按提及时间序，保留全部不同值）
    fact_bits: list[str] = []
    seen_facts: set = set()
    for m in members:
        for c in json.loads(m["claims_json"] or "[]"):
            if not isinstance(c, dict) or c.get("value") in (None, ""):
                continue
            bit = f"{c.get('predicate')}={c.get('value')}"
            if bit not in seen_facts:
                seen_facts.add(bit)
                fact_bits.append(bit)
    groups = {m["lineage_group"] or m["document_version_id"] for m in members}
    t = f"{(latest['event_time_lower'] or '?')[:10]}"
    if latest["event_time_upper"]:
        t += f"~{latest['event_time_upper'][:10]}"
    parts = [
        f"【{TYPE_LABELS.get(latest['event_type'], latest['event_type'])}】",
        f"主体：{'、'.join(actors[:4]) or '未知'}",
        f"对象：{'、'.join(objects[:4]) or '未知'}",
        f"动作：{latest['action'] or ''}",
        f"时间：{t}",
    ]
    if fact_bits:
        parts.append("关键事实：" + "；".join(fact_bits[:6]))
    parts.append(f"独立来源组：{len(groups)}；报道数：{len(members)}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 灰区 Jev 判别
# ---------------------------------------------------------------------------

def _llm_judge(conn, mention, ranked: list[dict], *, use_cache: bool = True) -> dict | None:
    """灰区判别（Jev 双问版）：
    - decision（choice）：候选1..N / new_event / related_process，附全分布 probabilities；
    - evidence_sufficient（noul）：信息是否足以判断（不足 → insufficient，宁待判勿猜）。

    输出映射回 v2 判别器留痕结构（decision / target_cluster_id / identity_matches /
    identity_conflicts / reason_code / missing_evidence），前端与审计兼容；
    身份锚点与冲突由规则推导（共享实体、类型、时间重叠 / 唯一编号、SKU、市场互斥）。
    """
    top = ranked[: cfg.load_config()["retrieval"]["judge_top_clusters"]]
    if not top:
        return None
    criteria = {}
    for i, cand in enumerate(top, 1):
        c = cand["cluster"]
        reps = _representatives(c)[:2]
        rep_txt = " / ".join(f"“{r.get('quote', '')[:60]}”" for r in reps) or "无"
        criteria[f"c{i}"] = (
            f"候选{i}（综合得分 {cand['score']}）：{c['summary']}｜类型 {c['event_type']}｜"
            f"时间 {c['event_time_lower'] or '?'}~{c['event_time_upper'] or '?'}｜"
            f"代表证据 {rep_txt}")
    criteria["new_event"] = "新提及与所有候选都不是同一事件，应新建事件"
    criteria["related_process"] = ("新提及与某候选主题相关（如同产品的价格历程、同一交易"
                                   "进程）但不是同一个原子事件")
    evs = json.loads(mention["evidence_json"] or "[]")[:2]
    state = {
        "新提及": {
            "框架": mention["frame_text"],
            "类型": mention["event_type"],
            "时间": f"{mention['event_time_lower'] or '?'}~{mention['event_time_upper'] or '?'}",
            "证据": [f"“{e.get('quote', '')[:80]}”" for e in evs] or ["无"],
        },
        "候选事件": {k: v for k, v in criteria.items() if k.startswith("c")},
        "判定规则": JEV_RULES,
    }
    questions = {
        "decision": {
            "type": "choice",
            "instructions": ("按 `判定规则` 判断 `新提及` 与哪个 `候选事件` 描述同一个现实"
                             "发生的原子事件；所有候选都不是时，从 new_event / "
                             "related_process 中选最贴切的一项。"),
            "criteria": criteria},
        "evidence_sufficient": {
            "type": "noul",
            "instructions": ("仅凭 `新提及` 与 `候选事件` 中的信息，是否足以可靠区分"
                             "「同一事件」与「新事件」（关键信息缺失时应判否）。"),
            "criteria": {
                "true": "参与方/对象/时间/动作信息足以做出可靠判断",
                "false": "关键信息缺失（如时间未知且对象相近），无法可靠判断"}},
    }
    # 判别路由：provider=jev 走此实现；本地学生训好后 config.judge.provider 切
    # cascade（本地快速通道+灰带升 Jev 终审），接入点见 ml/judges.py（接口同构，
    # 框架阶段保持 jev 直连以零回归）
    try:
        answers = llm.decide(state, questions, stage="judge", use_cache=use_cache)
    except llm.LLMError as e:
        observe.emit(conn, "resolve", f"判别器调用失败: {e}", level="error",
                     kind="resolve.judge_error", target_id=mention["mention_id"])
        return None
    a = answers.get("decision") or {}
    choice = a.get("choice")
    probs = a.get("probabilities") or {}
    suf = float((answers.get("evidence_sufficient") or {}).get("noul", 1.0))
    top_probs = {k: round(float(v), 4) for k, v in sorted(
        probs.items(), key=lambda kv: -kv[1])[:5]} if probs else {}

    judge = {"decision": "insufficient", "target_cluster_id": None,
             "identity_matches": [], "identity_conflicts": [],
             "reason_code": f"jev:{choice}", "missing_evidence": [],
             "jev_probabilities": top_probs}
    if suf < 0.5:
        judge["missing_evidence"] = [f"jev:evidence_sufficient={suf:.2f}"]
        return judge

    if isinstance(choice, str) and choice.startswith("c") and choice[1:].isdigit():
        idx = int(choice[1:])
        if 1 <= idx <= len(top):
            cand = top[idx - 1]
            judge["decision"] = "same_event"
            judge["target_cluster_id"] = cand["cluster_id"]
            judge["identity_matches"], judge["identity_conflicts"] = _identity_anchors(
                conn, mention, cand["cluster"])
    elif choice == "new_event":
        judge["decision"] = "new_event"
    elif choice == "related_process":
        judge["decision"] = "related_process"
        # 同过程挂靠目标：取判别分布中概率最高的候选，无分布时回退综合得分 top-1
        best = max(top_probs, key=lambda k: top_probs[k]) if any(
            k.startswith("c") for k in top_probs) else None
        if isinstance(best, str) and best.startswith("c") and best[1:].isdigit() \
                and 1 <= int(best[1:]) <= len(top):
            judge["target_cluster_id"] = top[int(best[1:]) - 1]["cluster_id"]
        else:
            judge["target_cluster_id"] = top[0]["cluster_id"]
    else:
        judge["missing_evidence"] = [f"jev:unrecognized_choice={choice}"]
    return judge


def _identity_anchors(conn, mention, cluster) -> tuple[list[str], list[str]]:
    """规则推导身份锚点/冲突（判别留痕用，不参与决策）。"""
    matches: list[str] = []
    conflicts: list[str] = []
    if cluster["event_type"] == mention["event_type"]:
        matches.append(f"同类型:{mention['event_type']}")
    m_ents = set(recall_mod._mention_entity_ids(mention))
    c_ents = _cluster_entity_ids(conn, cluster["cluster_id"])
    inter = m_ents & c_ents
    if inter:
        names = [r["canonical_name"] for r in conn.execute(
            f"SELECT canonical_name FROM entity WHERE entity_id IN "
            f"({','.join('?' * len(inter))})", tuple(sorted(inter))).fetchall()]
        matches.append("共享实体:" + "、".join(names[:3]))
    # 时间重叠（两边都已知才算锚点）
    m_lo = recall_mod._dt(mention["event_time_lower"])
    c_lo = recall_mod._dt(cluster["event_time_lower"])
    if m_lo and c_lo:
        m_hi = recall_mod._dt(mention["event_time_upper"]) or m_lo
        c_hi = recall_mod._dt(cluster["event_time_upper"]) or c_lo
        if m_lo <= c_hi and c_lo <= m_hi:
            matches.append("时间重叠")
        else:
            conflicts.append(f"时间不相交:{m_lo.date()}~{m_hi.date()} vs "
                             f"{c_lo.date()}~{c_hi.date()}")
    # 明确互斥（与 recall._hard_exclusion 同口径）
    m_scope = json.loads(mention["scope_json"] or "{}")
    c_scope = (json.loads(cluster["frame_json"]) or {}).get("scope") or {}
    for k, label in (("business_no", "唯一编号不同"), ("sku", "SKU不同"), ("market", "市场不同")):
        mv, cv = m_scope.get(k), c_scope.get(k)
        if mv and cv and str(mv).strip() != str(cv).strip():
            conflicts.append(f"{label}:{mv}≠{cv}")
    return matches, conflicts


def _resolve_target(judged: dict, ranked: list[dict]) -> str | None:
    """把判别器输出的 target_cluster_id 规范成候选集内的完整 ID。

    Jev 版输出已是完整 UUID（或 new_event/related_process 时为 None/最優候选 UUID），
    仍兼容 v2 三种形态：完整 UUID / UUID 前缀(≥8 位) / 候选序号("1"/"候选1")。
    """
    t = judged.get("target_cluster_id")
    if t is None:
        return None
    t = str(t).strip()
    if not t or t.lower() in ("null", "none"):
        return None
    ids = [c["cluster_id"] for c in ranked]
    if t in ids:
        return t
    # 前缀匹配
    if len(t) >= 8:
        hits = [i for i in ids if i.startswith(t)]
        if len(hits) == 1:
            return hits[0]
    # 序号匹配："1" / "候选1" / "candidate 1"
    num = "".join(ch for ch in t if ch.isdigit())
    if num.isdigit() and 1 <= int(num) <= len(ranked):
        return ranked[int(num) - 1]["cluster_id"]
    return None


# ---------------------------------------------------------------------------
# 决策落库：attach / create / pending
# ---------------------------------------------------------------------------

def _decision_row(conn, mention, action, target, ranked, rec, *, reason, judge=None,
                  scores=None) -> str:
    decision_id = det_uuid("dec", mention["mention_id"], action)
    conn.execute(
        "INSERT OR REPLACE INTO resolution_decision(decision_id, mention_id, action, "
        "target_cluster_id, candidates_json, scores_json, features_json, judge_json, "
        "reason_code, model_id, policy_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (decision_id, mention["mention_id"], action, target,
         # S3：留痕从 top10 扩到全池（可观测性——审计需区分"召回未相遇"与"排 11 名开外"；
         # 池上限 200，JSON 体积可控且 decision 表只增不改行为）
         json.dumps([{"cluster_id": c["cluster_id"], "score": c["score"],
                      "features": c["features"]} for c in ranked], ensure_ascii=False),
         json.dumps(scores or {c["cluster_id"]: c["score"] for c in ranked},
                    ensure_ascii=False),
         json.dumps({"coverage_ok": rec.get("coverage_ok"),
                     "channels": {k: len(v) for k, v in (rec.get("channels") or {}).items()},
                     "excluded": rec.get("excluded", [])[:10]}, ensure_ascii=False),
         json.dumps(judge, ensure_ascii=False) if judge else None,
         str(reason), cfg.load_config()["models"]["jev_model"], POLICY_VERSION, now_iso()))
    return decision_id


def _finish_attach(conn, mention, target_id, ranked, rec, *, action, reason, judge=None,
                   use_cache=True) -> dict:
    decision_id = _decision_row(conn, mention, action, target_id, ranked, rec,
                                reason=reason, judge=judge)
    _apply_membership(conn, mention, target_id, decision_id)
    cluster = _refresh_cluster(conn, target_id)
    _ensure_event_edges(conn, mention, cluster)
    _emit_decision(conn, mention, action, target_id, cluster, reason, ranked, judge)
    _enqueue_claim_jobs(conn, mention, cluster)
    return {"action": action, "cluster": target_id, "reason": reason}


def _finish_create(conn, mention, frame_vec, *, reason, scores, action, judge=None,
                   use_cache=True) -> dict:
    cluster_id = det_uuid("cluster", mention["mention_id"], mention["event_type"])
    decision_id = _decision_row(conn, mention, action, cluster_id, [],
                                {"coverage_ok": True, "channels": {}, "excluded": []},
                                reason=reason, judge=judge, scores=scores)
    now = now_iso()
    t_lo, t_hi = mention["event_time_lower"], mention["event_time_upper"]
    conn.execute(
        "INSERT OR IGNORE INTO event_cluster(cluster_id, version, event_type, state, process_id, "
        "frame_json, event_time_lower, event_time_upper, first_seen, last_seen, summary, "
        "centroid, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(cluster_id) DO UPDATE SET updated_at=excluded.updated_at",
        (cluster_id, 1, mention["event_type"], "provisional", None,
         json.dumps({"frame_text": mention["frame_text"],
                     "scope": json.loads(mention["scope_json"] or "{}"),
                     "phase": mention["event_phase"],
                     "representatives": _reps_from_mention(conn, mention)},
                    ensure_ascii=False),
         t_lo, t_hi, now, now, mention["frame_text"], None, now, now))
    _apply_membership(conn, mention, cluster_id, decision_id)
    _refresh_cluster(conn, cluster_id)
    cluster = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cluster_id,)).fetchone()
    _ensure_event_edges(conn, mention, cluster)
    _emit_decision(conn, mention, action, cluster_id, cluster, reason, [], judge)
    _enqueue_claim_jobs(conn, mention, cluster)
    return {"action": action, "cluster": cluster_id, "reason": reason}


def _finish_pending(conn, mention, ranked, rec, *, reason, judge=None) -> dict:
    _decision_row(conn, mention, "pending", None, ranked, rec, reason=reason, judge=judge)
    observe.emit(conn, "resolve", f"待判: {mention['frame_text'][:60]}（{reason}）",
                 level="warn", kind="resolve.pending", target_id=mention["mention_id"],
                 data={"reason": reason, "judge": judge})
    return {"action": "pending", "reason": reason}


def _apply_membership(conn, mention, cluster_id: str, decision_id: str) -> None:
    membership_id = det_uuid("mem", mention["mention_id"], cluster_id)
    conn.execute(
        "INSERT OR IGNORE INTO cluster_membership(membership_id, mention_id, cluster_id, "
        "decision_id, added_at) VALUES (?,?,?,?,?)",
        (membership_id, mention["mention_id"], cluster_id, decision_id, now_iso()))


def _reps_from_mention(conn, mention) -> list[dict]:
    dv = conn.execute(
        "SELECT dv.document_version_id, dv.title, d.canonical_url, sl.lineage_group, s.name "
        "FROM document_version dv JOIN document d ON d.document_id=dv.document_id "
        "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
        "LEFT JOIN source s ON s.source_id=d.source_id "
        "WHERE dv.document_version_id=?", (mention["document_version_id"],)).fetchone()
    reps = []
    for ev in json.loads(mention["evidence_json"] or "[]")[:1]:
        reps.append({"quote": ev.get("quote", ""), "document_version_id":
                     mention["document_version_id"], "source": dv["name"] if dv else None,
                     "url": dv["canonical_url"] if dv else None,
                     "lineage_group": dv["lineage_group"] if dv else None})
    return reps


def _refresh_cluster(conn, cluster_id: str):
    """重算簇的派生表示：成员、版本、摘要、代表证据、状态晋升、outbox。

    v3：无向量质心（centroid 恒 NULL）；事件卡文本仍生成（BM25 检索单元）。
    """
    conf = cfg.load_config()
    cluster = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cluster_id,)).fetchone()
    members = conn.execute(
        "SELECT m.*, cm.added_at, sl.lineage_group, dv.published_at, dv.document_version_id "
        "FROM cluster_membership cm JOIN event_mention m ON m.mention_id=cm.mention_id "
        "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
        "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
        "WHERE cm.cluster_id=? AND cm.removed_at IS NULL AND m.status='valid' "
        "ORDER BY COALESCE(dv.published_at, dv.created_at, cm.added_at), m.mention_id",
        (cluster_id,)).fetchall()
    if not members:
        return cluster
    new_version = cluster["version"] + 1
    latest = members[-1]
    t_lo = min([m["event_time_lower"] for m in members if m["event_time_lower"]] or [None],
               key=lambda x: x or "9999")
    t_hi = max([m["event_time_upper"] for m in members if m["event_time_upper"]] or [None],
               key=lambda x: x or "")
    # 代表证据：每来源组一条（覆盖不同来源，不机械取最新十条），最多 N 个
    seen_groups, reps = set(), []
    for m in members:
        lg = m["lineage_group"] or m["document_version_id"]
        if lg in seen_groups:
            continue
        seen_groups.add(lg)
        for ev in json.loads(m["evidence_json"] or "[]")[:1]:
            src = conn.execute(
                "SELECT s.name, d.canonical_url FROM document_version dv "
                "JOIN document d ON d.document_id=dv.document_id "
                "JOIN source s ON s.source_id=d.source_id "
                "WHERE dv.document_version_id=?", (m["document_version_id"],)).fetchone()
            reps.append({"quote": ev.get("quote", ""), "document_version_id":
                         m["document_version_id"], "source": src["name"] if src else None,
                         "url": src["canonical_url"] if src else None, "lineage_group": lg})
        if len(reps) >= conf["cluster"]["max_representatives"]:
            break
    frame = {"frame_text": latest["frame_text"],
             "scope": json.loads(latest["scope_json"] or "{}"),
             "phase": latest["event_phase"],
             "representatives": reps,
             "actors": json.loads(latest["actor_json"] or "[]"),
             "objects": json.loads(latest["object_json"] or "[]")}
    # 质心：来源组去权重的成员框架向量均值（质心只用于召回与 S_semantic 特征，不决定归并）
    vecs, used_groups = [], set()
    for m in members:
        lg = m["lineage_group"] or m["document_version_id"]
        if lg in used_groups:
            continue
        v = vectors.get_vector(conn, owner_type="mention", owner_id=m["mention_id"],
                               role="mention_frame")
        if v is not None:
            vecs.append(v)
            used_groups.add(lg)
    centroid = None
    if vecs:
        centroid = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
        vectors.add_vector(conn, role="cluster_centroid", owner_type="cluster",
                           owner_id=cluster_id, sub_id="", model_id=conf["models"]["embed_model"],
                           dim=conf["models"]["embed_dimensions"], vec=centroid,
                           text_hash=det_uuid("centroid", cluster_id, str(new_version)))
        # 代表证据向量复用成员证据向量（不重复调 embedding）
        for i, rep in enumerate(reps[:5]):
            ev_vec = vectors.get_vector(conn, owner_type="mention", owner_id=latest["mention_id"],
                                        role="evidence")
            if ev_vec is not None:
                vectors.add_vector(conn, role="cluster_rep", owner_type="cluster",
                                   owner_id=cluster_id, sub_id=f"rep{i}",
                                   model_id=conf["models"]["embed_model"],
                                   dim=conf["models"]["embed_dimensions"], vec=ev_vec,
                                   text_hash=det_uuid("rep", cluster_id, str(new_version), str(i)))
    # 状态晋升：独立来源组 ≥2 或成员 ≥3 → resolved
    state = cluster["state"]
    if state == "provisional" and (len(seen_groups) >= 2 or len(members) >= 3):
        state = "resolved"
    now = now_iso()
    card_text = build_event_card(members, conn)
    card_hash = sha256_text(card_text)
    conn.execute(
        "UPDATE event_cluster SET version=?, event_type=?, event_time_lower=?, "
        "event_time_upper=?, last_seen=?, summary=?, card_text=?, card_hash=?, frame_json=?, "
        "centroid=?, state=?, updated_at=? WHERE cluster_id=?",
        (new_version, latest["event_type"], t_lo or cluster["event_time_lower"],
         t_hi or cluster["event_time_upper"], now, latest["frame_text"], card_text, card_hash,
         json.dumps(frame, ensure_ascii=False),
         centroid.tobytes() if centroid is not None else None, state, now, cluster_id))
    # 事件卡向量：登记待嵌，run() 末尾批量补齐（归并召回只用质心，不需卡片向量；
    # 逐条单发 HTTP 是串行 apply 的主要耗时之一）
    _PENDING_CARD_EMBEDS.add(cluster_id)
    decision_id = conn.execute(
        "SELECT decision_id FROM cluster_membership WHERE cluster_id=? AND removed_at IS NULL "
        "ORDER BY added_at DESC LIMIT 1", (cluster_id,)).fetchone()["decision_id"]
    conn.execute(
        "INSERT OR REPLACE INTO cluster_version(cluster_id, version, decision_id, snapshot_json, "
        "recorded_at) VALUES (?,?,?,?,?)",
        (cluster_id, new_version, decision_id,
         json.dumps({"members": [m["mention_id"] for m in members], "summary": latest["frame_text"],
                     "state": state, "representatives": reps}, ensure_ascii=False), now))
    _outbox_event(conn, cluster_id, new_version, "event.changed",
                  {"cluster_id": cluster_id, "version": new_version, "type": latest["event_type"]})
    return conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cluster_id,)).fetchone()


def _ensure_event_edges(conn, mention, cluster) -> None:
    """图维护：participates_in（实体→事件）、part_of_process（事件→过程）、precedes（过程内先后）。"""
    ent_ids = recall_mod._mention_entity_ids(mention)
    for eid in ent_ids:
        _add_edge(conn, "entity", eid, "event", cluster["cluster_id"], "participates_in",
                  asserted_by="rule", note="自动建边：提及实体参与事件")
    # 过程：同对象 + 事件家族 → Process（两次降价不同事件、同一过程）
    family = {"price_change": "price_history", "product_launch": "product_lifecycle",
              "merger_deal": "deal_process"}.get(mention["event_type"], mention["event_type"])
    objs = [a.get("entity_id") for a in json.loads(mention["object_json"] or "[]")
            if isinstance(a, dict) and a.get("entity_id")]
    if not objs:
        objs = [a.get("entity_id") for a in json.loads(mention["actor_json"] or "[]")
                if isinstance(a, dict) and a.get("entity_id")]
    if objs:
        process_id = _get_or_create_process(conn, family, objs[0])
        if process_id and process_id != cluster["process_id"]:
            conn.execute("UPDATE event_cluster SET process_id=? WHERE cluster_id=?",
                         (process_id, cluster["cluster_id"]))
            _add_edge(conn, "event", cluster["cluster_id"], "process", process_id,
                      "part_of_process", asserted_by="rule", note=f"家族:{family}")
        if process_id:
            _update_precedes(conn, process_id)


PROCESS_LABELS = {"price_history": "价格历程", "product_lifecycle": "产品生命周期",
                  "deal_process": "交易进程"}


def _get_or_create_process(conn, family: str, object_entity_id: str) -> str | None:
    pid = det_uuid("proc", family, object_entity_id)
    ent = conn.execute("SELECT canonical_name FROM entity WHERE entity_id=?",
                       (object_entity_id,)).fetchone()
    if ent is None:
        return None
    title = f"{ent['canonical_name']}·{PROCESS_LABELS.get(family, family)}"
    conn.execute(
        "INSERT OR IGNORE INTO process(process_id, family, object_entity_id, title, created_at) "
        "VALUES (?,?,?,?,?)",
        (pid, family, object_entity_id, title, now_iso()))
    return pid


def _update_precedes(conn, process_id: str) -> None:
    """过程内按事件时间维护 precedes 边（只表先后，不表因果）。"""
    events = conn.execute(
        "SELECT cluster_id, event_time_lower FROM event_cluster WHERE process_id=? "
        "AND deleted_at IS NULL AND event_time_lower IS NOT NULL "
        "ORDER BY event_time_lower", (process_id,)).fetchall()
    for a, b in zip(events, events[1:]):
        if a["event_time_lower"] == b["event_time_lower"]:
            continue  # 同时间不排序，避免传递误并
        _add_edge(conn, "event", a["cluster_id"], "event", b["cluster_id"], "precedes",
                  asserted_by="rule", note="过程内时间先后")


def _link_related_process(conn, cluster_id: str, target_cluster_id: str) -> None:
    """judge 判定 related_process：新建事件挂到目标事件所在过程。"""
    target = conn.execute("SELECT process_id FROM event_cluster WHERE cluster_id=?",
                          (target_cluster_id,)).fetchone()
    if target and target["process_id"]:
        conn.execute("UPDATE event_cluster SET process_id=? WHERE cluster_id=?",
                     (target["process_id"], cluster_id))
        _add_edge(conn, "event", cluster_id, "process", target["process_id"], "part_of_process",
                  asserted_by="model", note="判别器:相关过程")
        _update_precedes(conn, target["process_id"])


_EDGE_RELS = {"participates_in", "part_of_process", "precedes", "follows", "supports",
              "contradicts", "corrects", "retracts", "related_to"}


def _add_edge(conn, from_type, from_id, to_type, to_id, relation, *, asserted_by="rule",
              note=None, evidence=None) -> str | None:
    assert relation in _EDGE_RELS
    if from_id == to_id and from_type == to_type:
        return None
    exist = conn.execute(
        "SELECT relation_id FROM semantic_relation WHERE from_type=? AND from_id=? AND to_type=? "
        "AND to_id=? AND relation=? AND deleted_at IS NULL",
        (from_type, from_id, to_type, to_id, relation)).fetchone()
    if exist:
        return exist["relation_id"]
    rid = det_uuid("rel", from_type, from_id, to_type, to_id, relation)
    cur = conn.execute(
        "INSERT OR IGNORE INTO semantic_relation(relation_id, from_type, from_id, to_type, to_id, "
        "relation, asserted_by, evidence_json, note, created_by, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (rid, from_type, from_id, to_type, to_id, relation, asserted_by,
         json.dumps(evidence or [], ensure_ascii=False), note, "pipeline", now_iso()))
    if cur.rowcount == 0:
        # 确定性 ID 撞上软删行 → 复活（同键即同逻辑边）
        conn.execute("UPDATE semantic_relation SET deleted_at=NULL WHERE relation_id=?", (rid,))
    return rid


def _outbox_event(conn, aggregate_id, version, event_type, payload: dict) -> None:
    from ..store import queue as Q
    oid = det_uuid("ob", aggregate_id, str(version), event_type)
    conn.execute(
        "INSERT OR IGNORE INTO outbox(outbox_id, aggregate_type, aggregate_id, aggregate_version, "
        "event_type, payload_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (oid, "event_cluster", aggregate_id, version, event_type,
         json.dumps(payload, ensure_ascii=False), now_iso()))
    conn.execute(
        "INSERT INTO projection_state(aggregate_type, aggregate_id, requested_version, "
        "indexed_version, updated_at) VALUES (?,?,?,0,?) "
        "ON CONFLICT(aggregate_type, aggregate_id) DO UPDATE SET "
        "requested_version=MAX(requested_version, excluded.requested_version), updated_at=excluded.updated_at",
        ("event_cluster", aggregate_id, version, now_iso()))
    Q.enqueue(conn, "intel.event.changed.v1",
              {"cluster_id": aggregate_id, "version": version,
               "idempotency_key": f"evt:{aggregate_id}:{version}"})


def _enqueue_claim_jobs(conn, mention, cluster) -> None:
    """归类完成后 → 事实更新流程（断言候选入队）。"""
    from ..store import queue as Q
    claims = json.loads(mention["claims_json"] or "[]")
    if not claims:
        return
    job_id = det_uuid("assertjob", mention["mention_id"])
    Q.enqueue(conn, "intel.assertion.candidate.v1",
              {"job_id": job_id, "mention_id": mention["mention_id"],
               "document_version_id": mention["document_version_id"],
               "event_id": cluster["cluster_id"],
               "idempotency_key": f"claims:{mention['mention_id']}"},
              trace_id=mention["mention_id"])


def _emit_decision(conn, mention, action, target_id, cluster, reason, ranked, judge) -> None:
    label = {"attach": "归入旧事件", "create_provisional": "新建候选事件",
             "judge_attach": "判别归并", "judge_create": "判别新建"}.get(action, action)
    top = ranked[0] if ranked else None
    observe.emit(
        conn, "resolve",
        f"{label}: {mention['frame_text'][:70]} → {target_id[:12] if target_id else ''}（{reason}）",
        kind="resolve.decision", target_id=mention["mention_id"],
        data={"action": action, "cluster_id": target_id, "reason": reason,
              "best_score": top["score"] if top else None,
              "cluster_state": cluster["state"] if cluster else None,
              "cluster_version": cluster["version"] if cluster else None,
              "cluster_summary": (cluster["summary"][:80] if cluster else None),
              "judge": judge})


def emit_recall_event(conn, mention, rec) -> None:
    observe.emit(conn, "recall",
                 f"召回: {len(rec['candidates'])} 候选 / 排除 {len(rec['excluded'])} / "
                 f"健康={rec['coverage_ok']}",
                 kind="recall.result", target_id=mention["mention_id"],
                 data={"candidates": rec["candidates"][:20],
                       "channels": {k: len(v) for k, v in rec["channels"].items()},
                       "excluded": rec["excluded"][:10], "coverage_ok": rec["coverage_ok"]})


def run(conn: sqlite3.Connection, *, use_cache: bool = True) -> dict:
    """按事件发生/发布时间顺序流式归类（模拟在线：只能用当时已到达的历史）。

    并行与可复现的共存方案（04 §10.2 串行键的时间片化）：
    1. 提及按发布日切时间片，片间屏障；
    2. 片开始时快照"已存在簇集合"，片内所有计划只能看到该快照
       （同一批并发到达的消息互不可见，天然消除竞争）；
    3. 片内按主对象实体分组，判别计划（只读 + Jev/缓存）线程池并行；
    4. 落库在主线程按片内原始顺序串行执行 —— 终态与串行版一致，重放可复现。
    """
    import sqlite3 as _sq
    rows = conn.execute(
        "SELECT m.mention_id, COALESCE(dv.published_at, dv.created_at, m.created_at) AS ts "
        "FROM event_mention m "
        "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
        "WHERE m.status='valid' AND NOT EXISTS ("
        "  SELECT 1 FROM cluster_membership cm WHERE cm.mention_id=m.mention_id "
        "  AND cm.removed_at IS NULL) "
        "ORDER BY (ts IS NULL), ts, m.mention_id"
    ).fetchall()
    stats = {"attach": 0, "judge_attach": 0, "create_provisional": 0, "judge_create": 0,
             "pending": 0, "skipped": 0}
    if not rows:
        conn.execute("UPDATE document_version SET status='resolved' WHERE status='extracted'")
        conn.commit()
        observe.emit(conn, "resolve", f"归类阶段完成: {stats}", kind="resolve.stage_done", data=stats)
        return stats
    conf = cfg.load_config()
    workers = max(1, int(conf.get("pipeline", {}).get("resolve_workers", 4)))
    db_path = cfg.DB_PATH

    def _plan_worker(mention_id: str, snapshot_ids: set):
        wconn = _sq.connect(str(db_path), timeout=30.0)
        wconn.row_factory = _sq.Row
        wconn.execute("PRAGMA busy_timeout=10000")
        try:
            return mention_id, resolve_plan(wconn, mention_id, use_cache=use_cache,
                                            allowed_cluster_ids=snapshot_ids)
        except Exception as e:  # noqa: BLE001 单条失败不阻断
            return mention_id, {"kind": "pending", "mention_id": mention_id, "ranked": [],
                                "rec": {"coverage_ok": False, "channels": {}, "excluded": []},
                                "reason": f"plan_error:{e}", "judge": None}
        finally:
            wconn.close()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 提及发布时间映射（apply 时增量维护簇的最早成员时间）
    ts_map = {r["mention_id"]: (r["ts"] or "") for r in rows}
    min_pub: dict[str, str] = {r["cluster_id"]: (r["mp"] or "")
                               for r in conn.execute("""
        SELECT c.cluster_id, MIN(COALESCE(dv.published_at, m.created_at)) mp
        FROM event_cluster c
        JOIN cluster_membership cm ON cm.cluster_id=c.cluster_id AND cm.removed_at IS NULL
        JOIN event_mention m ON m.mention_id=cm.mention_id AND m.status='valid'
        LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id
        WHERE c.deleted_at IS NULL AND c.state NOT IN ('redirected','split','deleted')
        GROUP BY c.cluster_id""").fetchall()}
    # 时间片切分（按天）
    slices: list[list[str]] = []
    cur_day, cur_list = None, []
    for r in rows:
        day = (r["ts"] or "")[:10]
        if day != cur_day:
            if cur_list:
                slices.append(cur_list)
            cur_day, cur_list = day, [r["mention_id"]]
        else:
            cur_list.append(r["mention_id"])
    if cur_list:
        slices.append(cur_list)

    for si, mention_ids in enumerate(slices):
        day = (conn.execute(
            "SELECT COALESCE(dv.published_at, m.created_at) ts FROM event_mention m "
            "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
            "WHERE m.mention_id=?", (mention_ids[0],)).fetchone() or {"ts": "?"})["ts"][:10]
        # 片快照（业务时间确定性，增量维护）：只含"最早成员发布日早于本片日期"的簇。
        # min_pub 映射随 apply 增量更新，O(1) 比对——替代每片一次的相关子查询
        # （千簇 × 千片时子查询是天级瓶颈）；语义与 SQL 版完全一致，可复现不变。
        day_start = day if day else "9999"  # 无日期片：仅收全日期簇之后的补充
        snapshot_ids = {cid for cid, mp in min_pub.items()
                        if (mp or "9999") < day_start}
        # 并行计划
        plans: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_plan_worker, mid, snapshot_ids) for mid in mention_ids]
            for fut in as_completed(futs):
                mid, plan = fut.result()
                plans[mid] = plan
        observe.emit(conn, "resolve", f"时间片 {day}：{len(mention_ids)} 条提及计划完成",
                     kind="resolve.slice", data={"day": day, "n": len(mention_ids)})
        # 主线程按原顺序落库
        for mid in mention_ids:
            out = apply_plan(conn, plans.get(mid, {"kind": "skipped"}), use_cache=use_cache)
            a = out.get("action")
            if a in stats:
                stats[a] += 1
            else:
                stats["skipped"] += 1
            cid = out.get("cluster")
            t = ts_map.get(mid)
            if cid and t:
                min_pub[cid] = min(min_pub.get(cid, t), t)
        conn.commit()  # 片级提交（崩溃只丢当前片，幂等重跑安全）
        flush_card_embeds(conn)
    flush_card_embeds(conn)  # 兜底：无日期片后仍有残留时
    conn.execute("UPDATE document_version SET status='resolved' WHERE status='extracted'")
    conn.commit()
    observe.emit(conn, "resolve", f"归类阶段完成: {stats}", kind="resolve.stage_done", data=stats)
    return stats
