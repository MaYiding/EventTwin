#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两代产线的高级模型二次评判（Jev 五维抽样评判，固定 seed 可复现）。

维度：
  A attach 精度   —— 判别并方向正确性（提及 vs 目标簇是否真同一事件）
  B 簇纯度        —— 多成员簇内一致性（成员 vs 簇摘要）
  C 过分裂率      —— 判别分方向正确性（同实体+同类型+时间重叠的簇对该并而未并）
  D 断言忠实度    —— 事实层（断言是否被原文证据支持）
  E 检索精度@5    —— 端到端（固定查询集，两库同算法检索后 Jev 判结果有用性）

用法：
  python3 compare/jev_judge.py <db_path> <out_json> [--queries-only]
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intel import config as cfg, llm  # noqa: E402

SEED = 20260929
RULES = (
    "同事件判定规则：原子事件=特定参与方+特定对象+特定时间+一次具体动作；同一动作被不同"
    "媒体报道（含转载）=同一事件；同一产品两次不同调价/发布=不同事件；并购宣布与交割=不同"
    "事件；主题相近但动作、对象或时间不同=不同事件；只按给出的信息判断，不按常识补充。")


def _quotes(js, n=2, w=80):
    try:
        arr = json.loads(js or "[]")
    except json.JSONDecodeError:
        arr = []
    out = []
    for e in arr[:n]:
        q = e.get("quote", "") if isinstance(e, dict) else str(e)
        if q:
            out.append(f"“{q[:w]}”")
    return out


def _reps(cluster_row, n=2):
    try:
        return _quotes(json.dumps(json.loads(cluster_row["frame_json"] or "{}")
                                  .get("representatives") or []), n)
    except (json.JSONDecodeError, KeyError):
        return []


# ---------------------------------------------------------------- 维度 A/B/C/D
def sample_and_questions(conn, which: str, n: int) -> dict:
    rng = random.Random(SEED)
    qs: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    def cluster(cid):
        return conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cid,)).fetchone()

    def mention(mid):
        return conn.execute("SELECT * FROM event_mention WHERE mention_id=?", (mid,)).fetchone()

    if which == "A":  # attach 精度
        rows = conn.execute(
            "SELECT mention_id, target_cluster_id FROM resolution_decision "
            "WHERE action IN ('attach','judge_attach') AND target_cluster_id IS NOT NULL"
        ).fetchall()
        pick = rng.sample(rows, min(n, len(rows)))
        for i, r in enumerate(pick):
            m, c = mention(r["mention_id"]), cluster(r["target_cluster_id"])
            if m is None or c is None:
                continue
            qs[f"a{i}"] = {
                "type": "noul",
                "instructions": (
                    f"按 `判定规则` 判断 `cases.a{i}.新提及` 与 `cases.a{i}.目标事件` 是否描述"
                    f"同一个现实发生的原子事件。"),
                "criteria": {"true": "同一事件（相同参与方/对象+时间+同一动作，含不同媒体报道）",
                             "false": "不同事件（动作/对象/时间不同，或仅主题相近）"}}
            meta[f"a{i}"] = {
                "新提及": {"框架": m["frame_text"],
                            "类型": m["event_type"],
                            "时间": f"{m['event_time_lower'] or '?'}~{m['event_time_upper'] or '?'}",
                            "证据": _quotes(m["evidence_json"]) or ["无"]},
                "目标事件": {"摘要": c["summary"],
                               "类型": c["event_type"],
                               "时间": f"{c['event_time_lower'] or '?'}~{c['event_time_upper'] or '?'}",
                               "代表证据": _reps(c) or ["无"]}}

    elif which == "B":  # 簇纯度
        clusters = conn.execute(
            """SELECT c.cluster_id FROM event_cluster c
               JOIN cluster_membership cm ON cm.cluster_id=c.cluster_id AND cm.removed_at IS NULL
               JOIN event_mention m ON m.mention_id=cm.mention_id AND m.status='valid'
               WHERE c.deleted_at IS NULL AND c.state='resolved'
               GROUP BY c.cluster_id HAVING COUNT(*)>=2""").fetchall()
        pick = rng.sample(clusters, min(20, len(clusters)))
        for i, cid in enumerate(pick):
            c = cluster(cid["cluster_id"])
            if c is None:
                continue
            members = conn.execute(
                """SELECT m.* FROM cluster_membership cm JOIN event_mention m
                   ON m.mention_id=cm.mention_id AND m.status='valid'
                   WHERE cm.cluster_id=? AND cm.removed_at IS NULL
                   ORDER BY cm.added_at""", (cid["cluster_id"],)).fetchall()
            sample_m = rng.sample(members, min(3, len(members)))
            for j, m in enumerate(sample_m):
                qid = f"b{i}_{j}"
                qs[qid] = {
                    "type": "noul",
                    "instructions": (f"按 `判定规则` 判断 `cases.{qid}.成员报道` 是否与 "
                                     f"`cases.{qid}.簇事件` 为同一事件。"),
                    "criteria": qs.get("b0_0", {}).get("criteria") or {
                        "true": "同一事件", "false": "不同事件（该成员被错误归入此簇）"}}
                meta[qid] = {
                    "簇事件": {"摘要": c["summary"],
                                 "类型": c["event_type"],
                                 "时间": f"{c['event_time_lower'] or '?'}~{c['event_time_upper'] or '?'}",
                                 "代表证据": _reps(c) or ["无"]},
                    "成员报道": {"框架": m["frame_text"],
                                  "证据": _quotes(m["evidence_json"], 1) or ["无"]}}

    elif which == "C":  # 过分裂：同实体+同类型+时间窗重叠的簇对
        pairs = conn.execute(
            """SELECT r1.to_id c1, r2.to_id c2 FROM semantic_relation r1
               JOIN semantic_relation r2 ON r1.from_id=r2.from_id AND r1.to_id<r2.to_id
               JOIN event_cluster k1 ON k1.cluster_id=r1.to_id
                    AND k1.deleted_at IS NULL AND k1.state NOT IN ('redirected','split','deleted')
               JOIN event_cluster k2 ON k2.cluster_id=r2.to_id
                    AND k2.deleted_at IS NULL AND k2.state NOT IN ('redirected','split','deleted')
               WHERE r1.relation='participates_in' AND r2.relation='participates_in'
                 AND r1.from_type='entity' AND r2.from_type='entity'
                 AND r1.deleted_at IS NULL AND r2.deleted_at IS NULL
                 AND k1.event_type=k2.event_type
                 AND MAX(k1.event_time_lower, k2.event_time_lower) <=
                     MIN(COALESCE(k1.event_time_upper,k1.event_time_lower),
                         COALESCE(k2.event_time_upper,k2.event_time_lower), '9999')
                 AND ABS(julianday(k1.event_time_lower)-julianday(k2.event_time_lower))<=30
               LIMIT 4000""").fetchall()
        pick = rng.sample(pairs, min(n, len(pairs))) if pairs else []
        for i, p in enumerate(pick):
            c1, c2 = cluster(p["c1"]), cluster(p["c2"])
            if c1 is None or c2 is None:
                continue
            qs[f"c{i}"] = {
                "type": "noul",
                "instructions": (f"按 `判定规则` 判断 `cases.c{i}.事件甲` 与 `cases.c{i}.事件乙` "
                                 f"是否为同一个现实发生的原子事件（若是，说明二者本应归入同一簇）。"),
                "criteria": {"true": "同一事件（分属两簇属过分裂）",
                             "false": "不同事件（两簇分开是正确的）"}}
            meta[f"c{i}"] = {
                "事件甲": {"摘要": c1["summary"], "类型": c1["event_type"],
                            "时间": f"{c1['event_time_lower'] or '?'}~{c1['event_time_upper'] or '?'}",
                            "代表证据": _reps(c1, 1) or ["无"]},
                "事件乙": {"摘要": c2["summary"], "类型": c2["event_type"],
                            "时间": f"{c2['event_time_lower'] or '?'}~{c2['event_time_upper'] or '?'}",
                            "代表证据": _reps(c2, 1) or ["无"]}}

    elif which == "D":  # 断言忠实度
        rows = conn.execute(
            """SELECT a.assertion_id, a.predicate, a.value_json, a.subject_entity_id,
                      a.document_version_id,
                      (SELECT quote FROM assertion_evidence e
                        WHERE e.assertion_id=a.assertion_id LIMIT 1) quote
               FROM assertion a ORDER BY a.assertion_id LIMIT 200000""").fetchall()
        pick = rng.sample(rows, min(n, len(rows)))
        for i, a in enumerate(pick):
            ent = conn.execute("SELECT canonical_name FROM entity WHERE entity_id=?",
                               (a["subject_entity_id"],)).fetchone()
            quote = a["quote"] or ""
            # 原文上下文：按 quote 定位前后 240 字
            ctx = ""
            if quote:
                dv = conn.execute(
                    "SELECT normalized_text FROM document_version WHERE document_version_id=?",
                    (a["document_version_id"],)).fetchone()
                if dv and dv["normalized_text"]:
                    pos = dv["normalized_text"].find(quote[:40])
                    if pos >= 0:
                        ctx = dv["normalized_text"][max(0, pos - 60):pos + len(quote) + 180]
            qs[f"d{i}"] = {
                "type": "noul",
                "instructions": (f"判断断言 `cases.d{i}.断言` 是否被 `cases.d{i}.原文上下文` "
                                 f"支持（值/主体相符；原文未提及则不支持）。"),
                "criteria": {"true": "原文明确支持该断言（值与主体相符）",
                             "false": "原文不支持（值不符/主体不符/未提及）"}}
            try:
                value = json.loads(a["value_json"])
            except (json.JSONDecodeError, TypeError):
                value = a["value_json"]
            meta[f"d{i}"] = {
                "断言": {"主体": ent["canonical_name"] if ent else a["subject_entity_id"],
                          "谓词": a["predicate"], "值": value},
                "原文上下文": ctx or quote or "（无引文）"}

    return {"questions": qs, "meta": meta}


def run_dimension(conn, which: str, n: int) -> dict:
    built = sample_and_questions(conn, which, n)
    qs, meta = built["questions"], built["meta"]
    if not qs:
        return {"n": 0}
    state = {"判定规则": RULES, "cases": meta}
    answers = llm.decide(state, qs, stage=f"judge2.{which}")
    scores = {k: float((v or {}).get("noul", 0.0)) for k, v in answers.items()}
    vals = list(scores.values())
    return {"n": len(vals),
            "mean": round(sum(vals) / len(vals), 4) if vals else None,
            "pos_rate": round(sum(1 for v in vals if v >= 0.5) / len(vals), 4) if vals else None,
            "scores": scores}


# ---------------------------------------------------------------- 维度 E 检索
QUERIES = [
    "小米YU7 上市价格",
    "特斯拉 Model 3 降价",
    "阿里巴巴 最新季度财报 营收",
    "字节跳动 TikTok 美国 出售",
    "华为 鸿蒙智行 新车发布",
    "苹果 Apple Intelligence 发布",
    "比亚迪 海外建厂 产能",
    "腾讯 游戏业务 收入变化",
    "百度 文心一言 大模型升级",
    "宁德时代 固态电池 进展",
    "美团 外卖 补贴大战",
    "小米汽车 召回",
]


def run_search(conn) -> dict:
    """检索精度（公平口径）：BM25 召回 + Jev 重排，两库同算子——只比库内容质量。

    不走 dense 通道：两库向量维度/模型不同（2560 vs 4096），且旧 embedding-4b 服务
    已退役无法为旧库重新向量化，混用算子会让对比失去控制变量。
    """
    from intel.query_service import _doc_index, _best_snippet
    per_query = {}
    for qi, query in enumerate(QUERIES):
        try:
            top_ids = _doc_index(conn).top(query, 30)
            rows = []
            for docv_id in top_ids:
                r = conn.execute(
                    "SELECT dv.document_version_id, dv.title, dv.normalized_text FROM document_version dv "
                    "WHERE dv.document_version_id=?", (docv_id,)).fetchone()
                if r is None:
                    continue
                rows.append({"title": r["title"],
                             "snippet": _best_snippet(r["normalized_text"] or "", query)})
            try:
                probs = llm.jev_choice_rank(query, [f"{it['title']} {it['snippet']}"
                                                    for it in rows], stage="judge2.E.rank")
                for i, it in enumerate(rows):
                    it["relevance"] = probs[i] if i < len(probs) else 0.0
                rows.sort(key=lambda x: -x.get("relevance", 0))
            except llm.LLMError:
                pass
            items = rows[:5]
        except Exception as e:  # noqa: BLE001
            per_query[query] = {"error": str(e)[:120]}
            continue
        if not items:
            per_query[query] = {"p5": None, "n": 0}
            continue
        qs, meta = {}, {}
        for i, it in enumerate(items):
            qid = f"e{qi}_{i}"
            qs[qid] = {
                "type": "noul",
                "instructions": (f"判断资料 `cases.{qid}.片段` 是否有助于回答查询"
                                 f"`cases.{qid}.查询`（直接包含答案信息才算有助于回答）。"),
                "criteria": {"true": "片段包含与查询直接相关的信息",
                             "false": "片段与查询无关或不含答案信息"}}
            meta[qid] = {"查询": query,
                         "片段": {"标题": it.get("title") or "", "摘录": it.get("snippet", "")[:200]}}
        answers = llm.decide({"判定规则": RULES, "cases": meta}, qs, stage="judge2.E")
        scores = [float((v or {}).get("noul", 0.0)) for v in answers.values()]
        per_query[query] = {
            "p5": round(sum(1 for s in scores if s >= 0.5) / len(scores), 3) if scores else None,
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
            "top1_title": items[0].get("title", "")[:40] if items else None}
    allp = [v["p5"] for v in per_query.values() if v.get("p5") is not None]
    return {"mAP5": round(sum(allp) / len(allp), 4) if allp else None,
            "per_query": per_query}


def main() -> None:
    db_path, out_path = sys.argv[1], sys.argv[2]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    out = {"db": db_path, "seed": SEED}
    if "--queries-only" not in sys.argv:
        out["A_attach_precision"] = run_dimension(conn, "A", 40)
        out["B_cluster_purity"] = run_dimension(conn, "B", 60)
        out["C_oversplit"] = run_dimension(conn, "C", 30)
        out["D_assertion_faithfulness"] = run_dimension(conn, "D", 30)
    out["E_search_precision"] = run_search(conn)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: ({kk: vv for kk, vv in v.items() if kk != "scores"}
                   if isinstance(v, dict) else v) for k, v in out.items()
               if k not in ("db",)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
