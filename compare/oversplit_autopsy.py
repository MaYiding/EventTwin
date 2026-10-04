#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""过分裂根因剖析：对"该并而未并"的簇对逐对追溯死因环节。

复现 jev_judge.py 维度 C 的抽样（同 seed），对 noul≥0.5 的簇对回溯归类留痕，
分类死因：
  R1 recall_miss   创建提及的候选池里根本没有对方簇（召回未相遇）
  R2 rank_gate     相遇了但综合分 < poc_score_new_floor(0.78) → 低分直通新建
  R3 hard_filter   被硬过滤排除（distinct_sku/market/phase/time_gap）
  R4 judge_wrong   进了灰区但判别器判了 new_event/insufficient
  R5 snapshot_iso  两簇首成员同发布日（时间片内互不可见，在线模拟固有）
  R6 other         其余（分数够但 bypass 未满足且未进灰区等边界）
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intel import llm, config as cfg  # noqa: E402

SEED = 20260929
RULES = ("同事件判定规则：原子事件=特定参与方+特定对象+特定时间+一次具体动作；同一动作被不同"
         "媒体报道（含转载）=同一事件；同一产品两次不同调价/发布=不同事件；主题相近但动作、"
         "对象或时间不同=不同事件；只按给出的信息判断，不按常识补充。")


def first_members(conn, cid):
    """簇的创建提及（首个成员）与全部成员提及（按落库序）。"""
    rows = conn.execute(
        """SELECT m.mention_id, m.frame_text, m.event_time_lower, m.event_time_upper,
                  dv.published_at, m.created_at
           FROM cluster_membership cm JOIN event_mention m ON m.mention_id=cm.mention_id
           LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id
           WHERE cm.cluster_id=? AND cm.removed_at IS NULL AND m.status='valid'
           ORDER BY cm.added_at""", (cid,)).fetchall()
    return rows


def decisions_of(conn, mention_id):
    return conn.execute(
        "SELECT action, reason_code, candidates_json, scores_json, features_json, judge_json "
        "FROM resolution_decision WHERE mention_id=? ORDER BY created_at", (mention_id,)).fetchall()


def autopsy_pair(conn, c1, c2) -> dict:
    """追溯：较晚簇的创建提及在归类时有没有机会见到较早簇。"""
    m1, m2 = first_members(conn, c1), first_members(conn, c2)
    if not m1 or not m2:
        return {"cause": "no_members"}
    t1 = (m1[0]["published_at"] or m1[0]["created_at"] or "")
    t2 = (m2[0]["published_at"] or m2[0]["created_at"] or "")
    late_first, early_first = (m2, m1) if t2 >= t1 else (m1, m2)
    late_cluster = c2 if t2 >= t1 else c1
    early_cluster = c1 if t2 >= t1 else c2
    out = {"first_pub_late": t2 if t2 >= t1 else t1, "same_day": t1[:10] == t2[:10]}

    # 较晚簇的每个早期成员的决策里找 early_cluster
    met_at = None          # 相遇的决策行
    for m in late_first[:3]:
        for d in decisions_of(conn, m["mention_id"]):
            cands = json.loads(d["candidates_json"] or "[]")
            if any(x["cluster_id"] == early_cluster for x in cands):
                met_at = (m["mention_id"], d)
                break
        if met_at:
            break
    if met_at is None:
        # 也查扩池复判（第二行决策）与排除列表
        excluded_hit = False
        for m in late_first[:3]:
            for d in decisions_of(conn, m["mention_id"]):
                feats = json.loads(d["features_json"] or "{}")
                for ex in (feats.get("excluded") or []):
                    if ex.get("cluster_id") == early_cluster:
                        excluded_hit = ex.get("reason")
        if excluded_hit:
            out["cause"] = f"R3_hard_filter:{excluded_hit}"
        elif out["same_day"]:
            out["cause"] = "R5_snapshot_iso"
        else:
            out["cause"] = "R1_recall_miss"
        return out
    mid, d = met_at
    out["met_action"] = d["action"]
    out["met_reason"] = d["reason_code"]
    # 相遇了：死在哪
    if d["action"] in ("create_provisional",):
        scores = json.loads(d["scores_json"] or "{}")
        top = max(scores.values()) if scores else 0.0
        out["top_score"] = round(top, 3)
        out["cause"] = "R2_rank_gate" if top < 0.78 else "R6_other_low_score_path"
    elif d["action"] == "judge_create":
        j = json.loads(d["judge_json"] or "{}")
        out["cause"] = f"R4_judge:{j.get('decision')}"
        out["jev_probs"] = j.get("jev_probabilities")
    elif d["action"] == "pending":
        out["cause"] = "R4_judge:insufficient_or_pending"
    elif d["action"] in ("attach", "judge_attach"):
        # 归并到了别的簇（错并到第三方）——不是过分裂的直接路径
        out["cause"] = "R6_attached_elsewhere"
        out["target"] = d["reason_code"]
    else:
        out["cause"] = "R6_other"
    return out


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "data/state/intel.db"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "compare/oversplit_autopsy_v31.json"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row

    # 复现 C 维度抽样
    rng = random.Random(SEED)
    pairs = conn.execute("""
        SELECT r1.to_id c1, r2.to_id c2 FROM semantic_relation r1
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
    pick = rng.sample(pairs, min(30, len(pairs)))

    # 复现 Jev 判定（同 seed 同题面 → 缓存命中，零成本）
    qs, meta = {}, {}
    for i, p in enumerate(pick):
        c1 = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (p["c1"],)).fetchone()
        c2 = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (p["c2"],)).fetchone()
        qs[f"c{i}"] = {"type": "noul",
                       "instructions": f"按 `判定规则` 判断 `cases.c{i}.事件甲` 与 `cases.c{i}.事件乙` 是否为同一个现实发生的原子事件（若是，说明二者本应归入同一簇）。",
                       "criteria": {"true": "同一事件（分属两簇属过分裂）", "false": "不同事件（两簇分开是正确的）"}}
        meta[f"c{i}"] = {"事件甲": {"摘要": c1["summary"], "类型": c1["event_type"],
                                    "时间": f"{c1['event_time_lower'] or '?'}~{c1['event_time_upper'] or '?'}"},
                          "事件乙": {"摘要": c2["summary"], "类型": c2["event_type"],
                                    "时间": f"{c2['event_time_lower'] or '?'}~{c2['event_time_upper'] or '?'}"}}
    ans = llm.decide({"判定规则": RULES, "cases": meta}, qs, stage="judge2.C")

    should_merge = [(i, p, float((ans.get(f"c{i}") or {}).get("noul", 0)))
                    for i, p in enumerate(pick)]
    should_merge = [x for x in should_merge if x[2] >= 0.5]
    print(f"该并簇对 {len(should_merge)}/30，逐对追溯死因…")

    results = []
    for i, p, score in should_merge:
        a = autopsy_pair(conn, p["c1"], p["c2"])
        a.update({"pair_idx": i, "jev_score": round(score, 2),
                  "s1": conn.execute("SELECT summary FROM event_cluster WHERE cluster_id=?",
                                     (p["c1"],)).fetchone()["summary"][:44],
                  "s2": conn.execute("SELECT summary FROM event_cluster WHERE cluster_id=?",
                                     (p["c2"],)).fetchone()["summary"][:44]})
        results.append(a)
        print(f"  [{a.get('cause'):24s}] {a['s1']} ⊕ {a['s2']}")

    from collections import Counter
    causes = Counter((r.get("cause") or "?").split(":")[0] for r in results)
    print("\n== 死因分布 ==")
    for k, v in causes.most_common():
        print(f"  {k}: {v}/{len(results)}")
    Path(out_path).write_text(json.dumps(
        {"n_should_merge": len(should_merge), "causes": dict(causes), "pairs": results},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("留档", out_path)


if __name__ == "__main__":
    main()
