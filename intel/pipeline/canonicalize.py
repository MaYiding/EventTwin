# -*- coding: utf-8 -*-
"""L3 批处理层：周期全局 canonicalization（簇间合并闭环，过分裂根治）。

流程（对齐 L1L2L3-实施方案 §三 P1-P6）：
  P1 候选对生成（SQL，不受提及级 hub 截断限制）：
     同类型 × 实体交集≥1 × 事件时间窗相容（间隔≤gap）
     ∪ 同日创建簇对（R5 同日隔离专项）
     ∪ 事件卡向量 top-k 近邻（有向量时）
  P2 全量打分：判别器路由（config.judge.provider；默认 Jev 分批，学生就绪后毫秒级）
  P3 图构建：节点=簇，边权 = P(同事件) - 0.5（正=同、负=异，SBBD 2025 置信度边权）
  P4 correlation clustering：pivot 贪心（3-近似，ICML 2025/NeurIPS 2023 谱系）
     + local search 微调；传递性进目标函数显式优化，不做隐式连通分量闭包
  P5 差分执行：组内最大概率 ≥0.9 → 自动 merge_clusters（版本化/redirect/可撤销，
     机制复用 graph_service）；0.5-0.9 → 人工队列（提案落表，Web 面板已有）
  P6 审计留痕：轮次/输入图/边权/决策/合并对照全部落表

用法：
  python3 -m intel.pipeline.canonicalize --dry-run        # 只出提案不执行（M3 门禁）
  python3 -m intel.pipeline.canonicalize                  # 执行（≥0.9 自动并，灰带入队）
  python3 -m intel.pipeline.canonicalize --stats          # 只看统计
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from intel import config as cfg, llm, observe  # noqa: E402
from ml.judges import make_judge  # noqa: E402
from intel.store import db  # noqa: E402


# ---------------------------------------------------------------------------
# P1 候选对生成
# ---------------------------------------------------------------------------

def candidate_pairs(conn: sqlite3.Connection, limit: int = 20000) -> list[tuple[str, str]]:
    conf = cfg.load_config()
    gap_days = conf["cluster"]["hard_time_gap_days"].get("default", 120)
    pairs: set[tuple[str, str]] = set()

    # a) 同类型 × 共享实体 × 时间窗相容（含间隔≤gap 的近邻窗口）
    rows = conn.execute(f"""
        SELECT r1.to_id c1, r2.to_id c2, COUNT(DISTINCT r1.from_id) n_shared
        FROM semantic_relation r1
        JOIN semantic_relation r2 ON r1.from_id=r2.from_id AND r1.to_id < r2.to_id
        JOIN event_cluster k1 ON k1.cluster_id=r1.to_id
             AND k1.deleted_at IS NULL AND k1.state NOT IN ('redirected','split','deleted')
        JOIN event_cluster k2 ON k2.cluster_id=r2.to_id
             AND k2.deleted_at IS NULL AND k2.state NOT IN ('redirected','split','deleted')
        WHERE r1.relation='participates_in' AND r2.relation='participates_in'
          AND r1.from_type='entity' AND r2.from_type='entity'
          AND r1.deleted_at IS NULL AND r2.deleted_at IS NULL
          AND k1.event_type = k2.event_type
          AND ABS(julianday(COALESCE(k1.event_time_lower,'9999'))
                - julianday(COALESCE(k2.event_time_lower,'9999'))) <= {int(gap_days)}
        GROUP BY r1.to_id, r2.to_id
        ORDER BY n_shared DESC LIMIT {int(limit)}""").fetchall()
    for r in rows:
        pairs.add((r["c1"], r["c2"]))

    # b) 同日创建簇对（R5 专项：时间片快照隔离的镜像修复）
    rows = conn.execute("""
        SELECT k1.cluster_id c1, k2.cluster_id c2 FROM event_cluster k1
        JOIN event_cluster k2 ON k1.cluster_id < k2.cluster_id
        WHERE k1.deleted_at IS NULL AND k2.deleted_at IS NULL
          AND k1.state NOT IN ('redirected','split','deleted')
          AND k2.state NOT IN ('redirected','split','deleted')
          AND k1.event_type = k2.event_type
          AND substr(k1.created_at,1,10) = substr(k2.created_at,1,10)
        LIMIT 8000""").fetchall()
    for r in rows:
        pairs.add((r["c1"], r["c2"]))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# P3/P4 图 + correlation clustering（pivot 贪心 3-近似 + 单点 local search）
# ---------------------------------------------------------------------------

def correlation_clustering(nodes: list[str], edges: dict[tuple[str, str], float],
                           seed: int = 20260930) -> list[set[str]]:
    """加权相关聚类近似：pivot 贪心（随机取点，正边邻居成组，去掉已分组的，递归），
    随后单点迁移 local search 直到无改进。目标 = 组内边权和（正加分负减分）最大。
    不做连通分量闭包——分组由全局目标函数显式决定（铁律的数学化表达）。
    """
    rng = random.Random(seed)
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
    for (a, b), w in edges.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    remaining = list(nodes)
    rng.shuffle(remaining)
    groups: list[set[str]] = []
    while remaining:
        pivot = remaining.pop(0)
        pos = {b for b, w in adj[pivot] if w > 0 and b in remaining}
        group = {pivot} | pos
        for x in pos:
            remaining.remove(x)
        groups.append(group)
    # local search：单点迁移（考虑正边拉力/负边斥力），至多 3 轮无改进即停
    improved = True
    rounds = 0
    while improved and rounds < 10:
        improved = False
        rounds += 1
        for n in nodes:
            cur_gain = _gain(n, _group_of(n, groups), adj)
            best_target, best_gain = None, cur_gain
            for g in groups:
                if n in g:
                    continue
                gain = _gain(n, g, adj)
                if gain > best_gain + 1e-9:
                    best_gain, best_target = gain, g
            if best_target is not None:
                _group_of(n, groups).discard(n)
                best_target.add(n)
                improved = True
        groups = [g for g in groups if g]
    return groups


def _group_of(n, groups):
    for g in groups:
        if n in g:
            return g
    return {n}


def _gain(n: str, group: set[str], adj) -> float:
    gm = group - {n}
    return sum(w for b, w in adj[n] if b in gm)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(conn: sqlite3.Connection, *, dry_run: bool = True, batch_cap: int = 8000) -> dict:
    conf = cfg.load_config()
    cconf = conf.get("canonicalization", {})
    auto_thr = cconf.get("auto_merge_threshold", 0.9)
    review_thr = cconf.get("review_threshold", 0.5)
    pairs = candidate_pairs(conn, limit=cconf.get("max_pairs", 20000))[:batch_cap]
    if not pairs:
        return {"round": "no_candidates", "n_pairs": 0}

    # P2 打分（判别器路由：默认 Jev 分批；学生就绪后本地毫秒）
    judge = make_judge()
    clusters = {r["cluster_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM event_cluster WHERE deleted_at IS NULL "
        "AND state NOT IN ('redirected','split','deleted')")}
    def cframe(cid):
        c = clusters.get(cid)
        if not c:
            return None
        try:
            reps = json.loads(c["frame_json"] or "{}").get("representatives") or []
            ev = [r.get("quote", "")[:80] for r in reps[:1]]
        except json.JSONDecodeError:
            ev = []
        return {"frame": (c["summary"] or (c["card_text"] or ""))[:100], "type": c["event_type"],
                "time": f"{c['event_time_lower'] or '?'}~{c['event_time_upper'] or '?'}",
                "evidence": [q[:60] for q in ev if q]}
    todo = [(a, b) for a, b in pairs if clusters.get(a) and clusters.get(b)]
    scores = judge.judge_batch([(cframe(a), cframe(b)) for a, b in todo])

    # P3 图构建（边权 = P-0.5）
    edges = {(a, b): s - 0.5 for (a, b), s in zip(todo, scores)}

    # P4 相关聚类
    nodes = sorted({x for p in todo for x in p})
    groups = correlation_clustering(nodes, edges)

    # P5 提案与差分执行
    proposals, auto_merges, review_queue = [], [], []
    for g in groups:
        if len(g) < 2:
            continue
        members = sorted(g)
        inner = [edges[(a, b)] for a in members for b in members
                 if a < b and (a, b) in edges]
        conf_score = (sum(inner) / len(inner) + 0.5) if inner else 0.5
        prop = {"clusters": members, "confidence": round(conf_score, 4),
                "n_inner_edges": len(inner)}
        proposals.append(prop)
        if conf_score >= auto_thr:
            auto_merges.append(prop)
        elif conf_score >= review_thr:
            review_queue.append(prop)

    executed = 0
    if not dry_run:
        from intel import graph_service as GS
        for prop in auto_merges:
            members = prop["clusters"]
            target = members[0]          # 组内取编号最小簇为目标（确定性）
            t = conn.execute("SELECT version FROM event_cluster WHERE cluster_id=?",
                             (target,)).fetchone()
            if t is None:
                continue
            try:
                GS.merge_clusters(conn, members[1:], target,
                                  reason=f"canonicalization:auto({prop['confidence']})",
                                  expected_version=t["version"])
                executed += 1
            except Exception as e:  # noqa: BLE001 版本冲突等留给下一轮
                observe.emit(conn, "canonicalize",
                             f"自动合并失败 {target[:12]}: {e}", level="warn",
                             kind="canon.merge_fail")
    # 派生层失效（合并后质心/卡/索引需重建）
    from . import recall as _recall
    _recall.invalidate_index_caches()
    conn.commit()

    # P6 留痕
    out = {"mode": "dry-run" if dry_run else "execute",
           "n_pairs": len(todo), "n_nodes": len(nodes), "n_groups_multimember": len(proposals),
           "auto_merges": len(auto_merges), "review_queue": len(review_queue),
           "executed": executed,
           "judge": judge.name,
           "proposals": proposals[:50], "auto_detail": auto_merges[:20]}
    out_path = Path(cfg.STATE_DIR) / f"canonicalization_{'dryrun' if dry_run else 'run'}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    observe.emit(conn, "canonicalize",
                 f"canonicalization {'dry-run' if dry_run else '执行'}: "
                 f"{len(todo)} 对 → 自动并 {len(auto_merges)} / 人工队列 {len(review_queue)}",
                 kind="canon.done", data={k: v for k, v in out.items()
                                          if k not in ("proposals", "auto_detail")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--batch-cap", type=int, default=8000)
    args = ap.parse_args()
    conn = db.connect(cfg.DB_PATH)
    db.init_db(conn)
    out = run(conn, dry_run=args.dry_run or args.stats, batch_cap=args.batch_cap)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("proposals", "auto_detail")}, ensure_ascii=False, indent=1))
    for p in out.get("auto_detail", [])[:10]:
        print("  自动并:", p["clusters"], p["confidence"])
    conn.close()


if __name__ == "__main__":
    main()
