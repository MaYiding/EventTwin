# -*- coding: utf-8 -*-
"""同事件判定金标 benchmark 构建（1000 对分层，Jev 预标 + 人工复核工作流）。

三层构成（hard 层决定区分度，是评测的核心）：
  pos  300 —— 明确同事件：多成员簇内对 + 转载对
  neg  400 —— 明确异事件：跨企业同类型对 + 跨类型对
  gray 300 —— 灰带：同实体×同类型×时间窗重叠的不同簇对（C 维度同款 SQL，最难层）

流程与状态：
  1) build   抽样 + Jev 预标（走缓存，重跑零成本）→ ml/benchmark/bm_v1.json
             （human 字段全 None = 待人工复核；jev_score 即 Jev 在本基准上的原始分）
  2) review  导出 ml/benchmark/review_v1.csv（对文本 + jev 分 + 复核列）供人工填写；
             填完 import 回填 → human ∈ {0,1}，金标升级为人工确认版
  3) evaluator.py 消费：human 存在时以 human 为准（防教师偏差泄漏），
             否则以 jev>=0.5 为临时标签并标注"未经人工复核"。

用法：
  python3 -m ml.benchmark build
  python3 -m ml.benchmark export   # 人工复核表
  python3 -m ml.benchmark import review_v1_filled.csv
"""
from __future__ import annotations

import csv
import json
import random
import sqlite3
from pathlib import Path

from intel import config as cfg
from .data_build import load_all, mention_frame, cluster_frame
from .judges import JevPairJudge, frame_text

BM_DIR = Path(__file__).parent / "benchmark"
SEED = 20260930
N_POS, N_NEG, N_GRAY = 300, 400, 300


def _sample(db: sqlite3.Connection, mentions, clusters, members, rng) -> list[dict]:
    pairs: list[dict] = []
    # pos：簇内对 + 转载对
    intra, cross_doc = [], []
    for cid, mids in members.items():
        if len(mids) < 2:
            continue
        take = mids if len(mids) <= 3 else rng.sample(mids, 3)
        for i in range(len(take)):
            for j in range(i + 1, len(take)):
                a, b = mention_frame(mentions[take[i]]), mention_frame(mentions[take[j]])
                if a["frame"] != b["frame"]:
                    intra.append({"a": a, "b": b, "layer": "pos", "origin": "intra_cluster"})
    for r in db.execute("""
            SELECT m1.mention_id a1, m2.mention_id a2 FROM event_mention m1
            JOIN event_mention m2 ON m1.mention_id < m2.mention_id
            JOIN document_version d1 ON d1.document_version_id=m1.document_version_id
            JOIN document_version d2 ON d2.document_version_id=m2.document_version_id
            JOIN source_lineage s1 ON s1.document_version_id=d1.document_version_id
            JOIN source_lineage s2 ON s2.document_version_id=d2.document_version_id
            WHERE s1.lineage_group=s2.lineage_group AND m1.status='valid' AND m2.status='valid'
              AND m1.frame_text != m2.frame_text LIMIT 2000"""):
        cross_doc.append({"a": mention_frame(mentions[r["a1"]]),
                          "b": mention_frame(mentions[r["a2"]]),
                          "layer": "pos", "origin": "lineage"})
    k_intra = int(N_POS * 0.7)
    pairs += rng.sample(intra, min(k_intra, len(intra)))
    pairs += rng.sample(cross_doc, min(N_POS - len([p for p in pairs if p["layer"] == "pos"]),
                                       len(cross_doc)))

    # neg：跨企业同类型 + 跨类型（时间也拉开）
    neg_same_type, neg_cross_type = [], []
    cids = list(clusters.keys())
    tries = 0
    while len(neg_same_type) + len(neg_cross_type) < N_NEG * 3 and tries < 30000:
        tries += 1
        c1, c2 = rng.sample(cids, 2)
        a, b = cluster_frame(clusters[c1]), cluster_frame(clusters[c2])
        ent1 = {r[0] for r in db.execute(
            "SELECT from_id FROM semantic_relation WHERE relation='participates_in' "
            "AND to_id=? AND deleted_at IS NULL", (c1,))}
        ent2 = {r[0] for r in db.execute(
            "SELECT from_id FROM semantic_relation WHERE relation='participates_in' "
            "AND to_id=? AND deleted_at IS NULL", (c2,))}
        if ent1 & ent2:
            continue  # 共享实体可能是同一事件，不放进明确负层
        if a["type"] == b["type"]:
            neg_same_type.append({"a": a, "b": b, "layer": "neg", "origin": "same_type"})
        elif rng.random() < 0.3:
            neg_cross_type.append({"a": a, "b": b, "layer": "neg", "origin": "cross_type"})
    pairs += rng.sample(neg_same_type, min(N_NEG // 2, len(neg_same_type)))
    pairs += rng.sample(neg_cross_type, min(N_NEG - len([p for p in pairs if p["layer"] == "neg"]),
                                            len(neg_cross_type)))

    # gray：同实体×同类型×时间窗重叠的不同簇对（最难层）
    gray = []
    for r in db.execute("""
            SELECT r1.to_id c1, r2.to_id c2 FROM semantic_relation r1
            JOIN semantic_relation r2 ON r1.from_id=r2.from_id AND r1.to_id<r2.to_id
            JOIN event_cluster k1 ON k1.cluster_id=r1.to_id AND k1.deleted_at IS NULL
                 AND k1.state NOT IN ('redirected','split','deleted')
            JOIN event_cluster k2 ON k2.cluster_id=r2.to_id AND k2.deleted_at IS NULL
                 AND k2.state NOT IN ('redirected','split','deleted')
            WHERE r1.relation='participates_in' AND r2.relation='participates_in'
              AND r1.deleted_at IS NULL AND r2.deleted_at IS NULL AND k1.event_type=k2.event_type
              AND ABS(julianday(k1.event_time_lower)-julianday(k2.event_time_lower))<=30
            LIMIT 5000"""):
        a, b = cluster_frame(clusters[r["c1"]]), cluster_frame(clusters[r["c2"]])
        gray.append({"a": a, "b": b, "layer": "gray", "origin": "hard_cluster"})
    pairs += rng.sample(gray, min(N_GRAY, len(gray)))
    return pairs


def build() -> Path:
    BM_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    db = sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    mentions, clusters, members = load_all(db)
    pairs = _sample(db, mentions, clusters, members, rng)
    db.close()
    print(f"抽样 {len(pairs)} 对，Jev 预标中（走缓存，重跑零成本）…")
    scores = JevPairJudge(stage="bm.prelabel").judge_batch([(p["a"], p["b"]) for p in pairs])
    out = {"version": "bm_v1", "seed": SEED,
           "human_reviewed": False, "created": __import__("datetime").datetime.now().isoformat(),
           "pairs": [{"id": i, "layer": p["layer"], "origin": p["origin"],
                      "a": frame_text(p["a"]), "b": frame_text(p["b"]),
                      "jev_score": round(s, 4), "human": None}
                     for i, (p, s) in enumerate(zip(pairs, scores))]}
    path = BM_DIR / "bm_v1.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    by_layer = {}
    for p in out["pairs"]:
        d = by_layer.setdefault(p["layer"], {"n": 0, "jev_pos": 0})
        d["n"] += 1
        d["jev_pos"] += 1 if p["jev_score"] >= 0.5 else 0
    print(json.dumps(by_layer, ensure_ascii=False))
    print(f"金标 v0（Jev 预标版）已写入 {path}；复核后升级人工确认版")
    return path


def export_review():
    bm = json.loads((BM_DIR / "bm_v1.json").read_text(encoding="utf-8"))
    path = BM_DIR / "review_v1.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "layer", "事件甲", "事件乙", "jev_score", "human(1=同事件,0=不同,留空=放弃)",
                    "备注"])
        for p in bm["pairs"]:
            w.writerow([p["id"], p["layer"], p["a"][:120], p["b"][:120], p["jev_score"], "", ""])
    print(f"复核表已导出 {path}（{len(bm['pairs'])} 行）")


def import_review(csv_path: str):
    bm = json.loads((BM_DIR / "bm_v1.json").read_text(encoding="utf-8"))
    n = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            h = (row.get("human(1=同事件,0=不同,留空=放弃)") or "").strip()
            if h in ("0", "1"):
                bm["pairs"][int(row["id"])]["human"] = int(h)
                n += 1
    bm["human_reviewed"] = n >= len(bm["pairs"]) * 0.8
    out = BM_DIR / ("bm_v1_human.json" if bm["human_reviewed"] else "bm_v1_partial.json")
    out.write_text(json.dumps(bm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"回填 {n} 条 → {out.name}（human_reviewed={bm['human_reviewed']}）")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "export":
        export_review()
    elif cmd == "import":
        import_review(sys.argv[2])
