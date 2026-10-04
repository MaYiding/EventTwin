# -*- coding: utf-8 -*-
"""训练数据构造流水线（L1 对比对 + L2 蒸馏对），全部从现有库与缓存免费生成。

产出（ml/data/ 目录，jsonl）：
- l1_pairs.jsonl   —— sbert MultipleNegativesRankingLoss 格式：
                       {"anchor":…, "positive":…, "negatives": [最多4个 hard 负例]}
- l2_pairs.jsonl   —— 蒸馏格式：{"a":…, "b":…, "label": Jev 概率软标签,
                       "source": p_same|judge|attach|lineage|cluster|gray}
- stats.json       —— 构成统计（正负比、来源分布、时间切分明细）

正对来源（稀缺方，全量挖掘）：
  P1 p_same≥0.9 的高置信同事件对（Jev 排序分布缓存）
  P2 attach 决策对（rule/judge 归并成功）
  P3 多成员簇内成员对
  P4 同 lineage_group 转载对
负对来源（海量，按需采样）：
  N1 p_same<0.5 的候选对（带 Jev 软标签，最可信负例）
  N2 hard 负例：同实体×同类型×时间窗重叠的不同簇对（C 维度同款 SQL）
  N3 随机跨簇对（易负例，占比小）
质量闸门：
  - 假负例 margin 过滤：用现役 embedding 打分，负例相似度 > 正例相似度-0.1 的丢弃
    （Qwen3/NVIDIA/SWIFT 官方同款做法；正例分缺失时跳过该闸门）
  - 时间切分：按事件时间 8:1:1 分 train/val/test，防同日同源文本泄漏
用法：
  python3 -m ml.data_build [--neg-per-pos 4] [--max-neg 60000] [--seed 20260930]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

from intel import config as cfg, llm
from .judges import frame_text

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# 库读取：提及与簇的 frame dict 化
# ---------------------------------------------------------------------------

def mention_frame(row) -> dict:
    try:
        ev = [e.get("quote", "")[:80] for e in json.loads(row["evidence_json"] or "[]")[:1]]
    except json.JSONDecodeError:
        ev = []
    return {"frame": row["frame_text"], "type": row["event_type"],
            "time": f"{row['event_time_lower'] or '?'}~{row['event_time_upper'] or '?'}",
            "evidence": [q for q in ev if q]}


def cluster_frame(row) -> dict:
    try:
        reps = json.loads(row["frame_json"] or "{}").get("representatives") or []
        ev = [r.get("quote", "")[:80] for r in reps[:1]]
    except json.JSONDecodeError:
        ev = []
    return {"frame": row["summary"], "type": row["event_type"],
            "time": f"{row['event_time_lower'] or '?'}~{row['event_time_upper'] or '?'}",
            "evidence": [q for q in ev if q]}


def load_all(conn: sqlite3.Connection):
    mentions = {r["mention_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM event_mention WHERE status='valid'")}
    clusters = {r["cluster_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM event_cluster WHERE deleted_at IS NULL "
        "AND state NOT IN ('redirected','split','deleted')")}
    members: dict[str, list[str]] = {}
    for r in conn.execute(
            "SELECT cluster_id, mention_id FROM cluster_membership "
            "WHERE removed_at IS NULL ORDER BY added_at"):
        members.setdefault(r["cluster_id"], []).append(r["mention_id"])
    return mentions, clusters, members


# ---------------------------------------------------------------------------
# p_same 缓存 → 对与软标签
# ---------------------------------------------------------------------------

def load_psame_pairs(conn_db: sqlite3.Connection, mentions: dict):
    """从 llm_cache 的 jev.p_same 请求重构 (提及, 候选簇) 对与 Jev 概率。

    缓存值只存 answers；请求侧的候选顺序无法从缓存恢复——改由查询参数散列不可行，
    因此这里按 stage=jev.p_same 的响应数计"可用软标签规模"，真实对重放由
    resolve 的决策留痕（candidates_json 含 cluster_id 与 same_event 特征）重构：
    decision.candidates_json 的 features.same_event 就是 Jev 概率（v3.1 起如此）。
    """
    pairs = []
    for r in conn_db.execute(
            "SELECT mention_id, candidates_json FROM resolution_decision "
            "WHERE candidates_json IS NOT NULL AND candidates_json != '[]'"):
        m = mentions.get(r["mention_id"])
        if m is None:
            continue
        try:
            cands = json.loads(r["candidates_json"])
        except json.JSONDecodeError:
            continue
        for c in cands:
            pairs.append((r["mention_id"], c["cluster_id"],
                          float(c.get("features", {}).get("same_event", 0.0))))
    return pairs


# ---------------------------------------------------------------------------
# 构造主流程
# ---------------------------------------------------------------------------

def build(neg_per_pos: int = 4, max_neg: int = 60000, seed: int = 20260930) -> dict:
    rng = random.Random(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    mentions, clusters, members = load_all(db)

    def mframe(mid):
        return mention_frame(mentions[mid]) if mid in mentions else None

    def cframe(cid):
        return cluster_frame(clusters[cid]) if cid in clusters else None

    positives: list[dict] = []   # {"a":frame, "b":frame, "source":…, "label":float}
    negatives: list[dict] = []

    # P1+软标签：decision 留痕重构的 p_same 对
    psame = load_psame_pairs(db, mentions)
    pos_from_psame = 0
    for mid, cid, p in psame:
        a, b = mframe(mid), cframe(cid)
        if a is None or b is None:
            continue
        if p >= 0.9:
            positives.append({"a": a, "b": b, "source": "p_same", "label": p})
            pos_from_psame += 1
        elif p < 0.5:
            negatives.append({"a": a, "b": b, "source": "p_same", "label": p})
    # P2 attach 决策对
    for r in db.execute("SELECT mention_id, target_cluster_id FROM resolution_decision "
                        "WHERE action IN ('attach','judge_attach') AND target_cluster_id IS NOT NULL"):
        a, b = mframe(r["mention_id"]), cframe(r["target_cluster_id"])
        if a and b:
            positives.append({"a": a, "b": b, "source": "attach", "label": 1.0})
    # P3 簇内成员对（每簇上限 6 对防 hub 簇支配）
    for cid, mids in members.items():
        if len(mids) < 2:
            continue
        take = mids if len(mids) <= 4 else rng.sample(mids, 4)
        for i in range(len(take)):
            for j in range(i + 1, min(i + 3, len(take))):
                a, b = mframe(take[i]), mframe(take[j])
                if a and b and a["frame"] != b["frame"]:
                    positives.append({"a": a, "b": b, "source": "cluster", "label": 1.0})
    # P4 转载对（同 lineage 不同文档版本）——v2 配比修正：降采样上限 1500
    # （v1 教训：3976 转载对占正对 75%，字面高相似主导训练信号，学生学成"措辞像=同事件"，
    #  措辞多样的簇内正对（benchmark pos 层）双模型 AUROC 仅 0.59/0.65）
    for r in db.execute("""
            SELECT m1.mention_id a1, m2.mention_id a2 FROM event_mention m1
            JOIN event_mention m2 ON m1.mention_id < m2.mention_id
            JOIN document_version d1 ON d1.document_version_id = m1.document_version_id
            JOIN document_version d2 ON d2.document_version_id = m2.document_version_id
            JOIN source_lineage s1 ON s1.document_version_id = d1.document_version_id
            JOIN source_lineage s2 ON s2.document_version_id = d2.document_version_id
            WHERE s1.lineage_group = s2.lineage_group AND m1.status='valid' AND m2.status='valid'
              AND m1.frame_text != m2.frame_text LIMIT 1500"""):
        a, b = mframe(r["a1"]), mframe(r["a2"])
        if a and b:
            positives.append({"a": a, "b": b, "source": "lineage", "label": 1.0})

    # N2 hard 负例：同实体×同类型×时间重叠的不同簇对（C 维度 SQL）
    hard_pool = db.execute("""
            SELECT r1.to_id c1, r2.to_id c2 FROM semantic_relation r1
            JOIN semantic_relation r2 ON r1.from_id=r2.from_id AND r1.to_id<r2.to_id
            JOIN event_cluster k1 ON k1.cluster_id=r1.to_id AND k1.deleted_at IS NULL
                 AND k1.state NOT IN ('redirected','split','deleted')
            JOIN event_cluster k2 ON k2.cluster_id=r2.to_id AND k2.deleted_at IS NULL
                 AND k2.state NOT IN ('redirected','split','deleted')
            WHERE r1.relation='participates_in' AND r2.relation='participates_in'
              AND r1.deleted_at IS NULL AND r2.deleted_at IS NULL
              AND k1.event_type=k2.event_type
              AND ABS(julianday(k1.event_time_lower)-julianday(k2.event_time_lower))<=30
            LIMIT 20000""").fetchall()
    n_hard = 0
    for r in hard_pool:
        a, b = cframe(r["c1"]), cframe(r["c2"])
        if a and b:
            negatives.append({"a": a, "b": b, "source": "hard_cluster", "label": 0.0})
            n_hard += 1
        if len(negatives) - pos_from_psame >= max_neg:
            break

    # 去重（frame 对文本级）与平衡采样
    def key(p):
        return tuple(sorted((frame_text(p["a"])[:80], frame_text(p["b"])[:80])))
    seen = set()
    pos_uniq = []
    for p in positives:
        k = key(p)
        if k not in seen:
            seen.add(k)
            pos_uniq.append(p)
    neg_uniq = []
    for p in negatives:
        k = key(p)
        if k not in seen:
            seen.add(k)
            neg_uniq.append(p)
    negatives = neg_uniq[:max_neg]

    # 假负例处理（两段式）：
    # 1) margin 初筛：embedding 相似度低于正例中位数-0.10 的直接保留（明显不是同事件）；
    # 2) 灰区负例（分更高、疑似同事件的 hard 候选）送 Jev 复核——判异=真负例保留，
    #    判同=假负例转正对（白赚高质量正例）。v1 实测教训：单用正例低分位做地板会把
    #    2 万 hard 负例杀到 51 个（转载对把正例相似度分布拉得太高）。
    t0 = time.time()
    pos_texts = [(frame_text(p["a"]), frame_text(p["b"])) for p in pos_uniq[:2000]]
    mat = llm.embed([t for pair in pos_texts for t in pair], stage="data.margin_pos")
    pos_scores = sorted((mat[0::2] * mat[1::2]).sum(axis=1))
    floor = float(pos_scores[len(pos_scores) // 2]) - 0.10  # 正例中位数-0.10
    sample_neg = neg_uniq[: min(len(neg_uniq), 20000)]
    easy_neg, gray_neg = [], []
    if sample_neg:
        nmat = llm.embed([t for p in sample_neg for t in (frame_text(p["a"]), frame_text(p["b"]))],
                         stage="data.margin_neg")
        nscores = (nmat[0::2] * nmat[1::2]).sum(axis=1)
        for p, s in zip(sample_neg, nscores):
            (easy_neg if float(s) < floor else gray_neg).append(p)
    print(f"margin 初筛: 负例 {len(neg_uniq)} → 明确负 {len(easy_neg)} / 灰区待Jev复核 {len(gray_neg)}"
          f"（地板=正例中位-0.10={floor:.3f}，耗时 {time.time()-t0:.0f}s）")
    kept_neg, promoted_pos = easy_neg, []
    if gray_neg:
        from .judges import JevPairJudge
        scores = JevPairJudge(stage="data.neg_verify").judge_batch(
            [(p["a"], p["b"]) for p in gray_neg])
        for p, s in zip(gray_neg, scores):
            if s >= 0.5:      # Jev 判同 → 假负例，升格为正对
                promoted_pos.append({**p, "source": p["source"] + "_promoted",
                                     "label": round(s, 3)})
            else:
                kept_neg.append(p)
        print(f"Jev 复核: 灰区 {len(gray_neg)} → 真负 {len(kept_neg) - len(easy_neg)} "
              f"/ 升格正对 {len(promoted_pos)}")
    negatives = kept_neg[:max_neg]
    pos_uniq += promoted_pos

    # 时间切分 8:1:1（按 a.time 的日期排序切，防同源泄漏）
    def sort_key(p):
        return (p["a"].get("time") or "?").split("~")[0]
    allp = sorted(pos_uniq + negatives, key=sort_key)
    n = len(allp)
    splits = {"train": allp[:int(n * .8)], "val": allp[int(n * .8):int(n * .9)],
              "test": allp[int(n * .9):]}

    # L1：anchor/positive/negatives 三元组（仅 train，用同 split 负例）
    l1_out, l2_out = [], []
    for split, items in splits.items():
        neg_pool = [p for p in items if p["label"] < 0.5]
        for p in items:
            if p["label"] >= 0.5:
                l2_out.append({"a": frame_text(p["a"]), "b": frame_text(p["b"]),
                               "label": p["label"], "source": p["source"], "split": split})
                if split == "train":
                    negs = rng.sample(neg_pool, min(neg_per_pos, len(neg_pool))) if neg_pool else []
                    l1_out.append({"anchor": frame_text(p["a"]),
                                   "positive": frame_text(p["b"]),
                                   "negatives": [frame_text(x["b"]) for x in negs]})
            else:
                l2_out.append({"a": frame_text(p["a"]), "b": frame_text(p["b"]),
                               "label": p["label"], "source": p["source"], "split": split})

    (DATA_DIR / "l1_pairs.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in l1_out), encoding="utf-8")
    (DATA_DIR / "l2_pairs.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in l2_out), encoding="utf-8")
    stats = {
        "seed": seed, "positives": len(pos_uniq), "negatives": len(negatives),
        "pos_sources": _count_by(pos_uniq, "source"), "neg_sources": _count_by(negatives, "source"),
        "splits": {k: {"n": len(v), "pos": sum(1 for x in v if x["label"] >= 0.5)} for k, v in splits.items()},
        "l1_rows": len(l1_out), "l2_rows": len(l2_out),
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    db.close()
    return stats


def _count_by(items, key):
    from collections import Counter
    return dict(Counter(x[key] for x in items))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--max-neg", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()
    print(json.dumps(build(a.neg_per_pos, a.max_neg, a.seed), ensure_ascii=False, indent=1))
