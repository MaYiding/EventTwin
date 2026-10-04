#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练数据大扩产 v3（用户指示：不人工打标，Qwen3.8-Max 仲裁灰带；体量上一个数量级）。

三通道对池（目标 15-20 万对，正对 2-3 万）：
  C1 簇×簇·同类型+共享实体+时间窗（canonicalize 同款，扩到 6 万对）
     —— Jev 打分（走缓存，canonicalize 两轮已覆盖 2.8 万对）
  C2 簇×簇·同类型+不共享实体+时间窗 ≤30d（新 hard 带：同类型不同主体的难负例）
  C3 提及×簇·decision 留痕全量（16.7 万对 p_same 分数现成，零成本）
  C4 提及×提及·同簇（正对扩产）+ 转载对 + attach 对（原通道保留）

标签三闸门：
  1) Jev ≥0.9 / ≤0.1 → 直接正/负（高置信带）
  2) Jev ∈[0.3,0.7] 灰带 → **qwen3.8-max-0902 仲裁**（10 对/请求批量；
     一致→确认；不一致→Max 复判一次，再不一致以 Max 为准——用户指定 Max 为准确打标器）
  3) 中间带（0.1-0.3 / 0.7-0.9）→ 保留 Jev 软标签（软标签蒸馏本就吃概率）

产出（覆盖 ml/data/）：l1_pairs.jsonl / l2_pairs.jsonl / stats.json（带 v3 标记）
用法：python3 -m ml.data_build_v3 [--max-pairs 200000] [--no-arbitrate(调试)]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import time
from pathlib import Path

from intel import config as cfg, llm
from ml.judges import frame_text

DATA_DIR = Path(__file__).parent / "data"
SEED = 20261002
MAX_MODEL = "qwen/qwen3.8-max-0902"


def _mention_frame(r) -> dict:
    try:
        ev = [e.get("quote", "")[:80] for e in json.loads(r["evidence_json"] or "[]")[:1]]
    except json.JSONDecodeError:
        ev = []
    return {"frame": r["frame_text"], "type": r["event_type"],
            "time": f"{r['event_time_lower'] or '?'}~{r['event_time_upper'] or '?'}",
            "evidence": [q for q in ev if q]}


def _cluster_frame(r) -> dict:
    try:
        reps = json.loads(r["frame_json"] or "{}").get("representatives") or []
        ev = [x.get("quote", "")[:60] for x in reps[:1]]
    except json.JSONDecodeError:
        ev = []
    return {"frame": (r["summary"] or (r["card_text"] or ""))[:100], "type": r["event_type"],
            "time": f"{r['event_time_lower'] or '?'}~{r['event_time_upper'] or '?'}",
            "evidence": [q for q in ev if q]}


def max_arbitrate(pairs, scores_in, key_prefix="arb"):
    """qwen3.8-max-0902 批量仲裁。返回 {idx: max_score}。10 对/请求。"""
    import requests
    key = llm._load_env() or None
    import os
    KEY = os.environ.get("OR_KEY")
    RULES = ("同事件判定规则：原子事件=特定参与方+特定对象+特定时间+一次具体动作；同一动作被不同"
             "媒体报道（含转载）=同一事件；同一产品先后两次不同调价/发布=不同事件；并购宣布与交割"
             "=不同事件；旧闻被重新报道=同一事件；主题相近但动作/对象/时间不同=不同事件；只按给出的"
             "信息判断。")
    FMT = '{"i": 对编号, "same": 0到100整数}'
    out = {}
    t0 = time.time()
    B = 10
    items = list(enumerate(pairs))
    for bi in range(0, len(items), B):
        chunk = items[bi:bi + B]
        lines = [f"【对{j+1}】事件甲：{frame_text(p['a'])[:150]}\n事件乙：{frame_text(p['b'])[:150]}"
                 for j, (idx, p) in enumerate(chunk)]
        body = {"model": MAX_MODEL, "temperature": 0.0, "max_tokens": 3000,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content":
                    f"{RULES}\n\n判断每一对事件是否同一个现实发生的原子事件。只输出JSON数组，"
                    f"每项形如 {FMT}：\n\n" + "\n\n".join(lines)}]}
        got = {}
        for attempt in range(3):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=150,
                                  headers={"Authorization": f"Bearer {KEY}"}, json=body)
                r.raise_for_status()
                c = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\[.*\]", c, re.S)
                if not m:
                    raise ValueError("no json")
                got = {int(x["i"]) - 1: int(x["same"]) / 100 for x in json.loads(m.group())}
                break
            except Exception:
                time.sleep(3)
        for j, (idx, _) in enumerate(chunk):
            if j in got:
                out[idx] = got[j]
        if (bi // B + 1) % 20 == 0:
            print(f"  Max仲裁 {bi+B}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def build(max_pairs=200000, arbitrate=True) -> dict:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    clusters = {r["cluster_id"]: dict(r) for r in db.execute(
        "SELECT * FROM event_cluster WHERE deleted_at IS NULL "
        "AND state NOT IN ('redirected','split','deleted')")}
    mentions = {r["mention_id"]: dict(r) for r in db.execute(
        "SELECT * FROM event_mention WHERE status='valid'")}
    members = {}
    for r in db.execute("SELECT cluster_id, mention_id FROM cluster_membership "
                        "WHERE removed_at IS NULL ORDER BY added_at"):
        members.setdefault(r["cluster_id"], []).append(r["mention_id"])

    all_pairs: list[dict] = []   # {"a","b","label","source"}  label=None 待定

    # ---- C3：decision 留痕全量（提及×簇，p_same 分数现成，零成本） ----
    n_c3 = 0
    for r in db.execute("SELECT mention_id, candidates_json FROM resolution_decision "
                        "WHERE candidates_json IS NOT NULL AND candidates_json!='[]'"):
        m = mentions.get(r["mention_id"])
        if not m:
            continue
        try:
            cands = json.loads(r["candidates_json"])
        except json.JSONDecodeError:
            continue
        fa = _mention_frame(m)
        for c in cands:
            k = clusters.get(c["cluster_id"])
            if not k:
                continue
            p = float(c.get("features", {}).get("same_event", 0.0))
            all_pairs.append({"a": fa, "b": _cluster_frame(k), "label": p,
                              "source": "c3_psame"})
            n_c3 += 1

    # ---- C1：簇×簇 同类型+共享实体+时间窗（canonicalize 同款，Jev 走缓存） ----
    jev = llm.jev_choice_rank if False else None
    from ml.judges import JevPairJudge
    judge = JevPairJudge(stage="data.v3.cluster")
    c1_rows = db.execute("""
        SELECT r1.to_id c1, r2.to_id c2 FROM semantic_relation r1
        JOIN semantic_relation r2 ON r1.from_id=r2.from_id AND r1.to_id<r2.to_id
        JOIN event_cluster k1 ON k1.cluster_id=r1.to_id AND k1.deleted_at IS NULL
             AND k1.state NOT IN ('redirected','split','deleted')
        JOIN event_cluster k2 ON k2.cluster_id=r2.to_id AND k2.deleted_at IS NULL
             AND k2.state NOT IN ('redirected','split','deleted')
        WHERE r1.relation='participates_in' AND r2.relation='participates_in'
          AND r1.deleted_at IS NULL AND r2.deleted_at IS NULL AND k1.event_type=k2.event_type
          AND ABS(julianday(COALESCE(k1.event_time_lower,'9999'))
                - julianday(COALESCE(k2.event_time_lower,'9999'))) <= 120
        LIMIT 60000""").fetchall()
    todo = [(r["c1"], r["c2"]) for r in c1_rows
            if r["c1"] in clusters and r["c2"] in clusters]
    print(f"C1 簇对 {len(todo)} → Jev 打分（缓存命中部分免费）…", flush=True)
    scores_c1 = judge.judge_batch([(_cluster_frame(clusters[a]), _cluster_frame(clusters[b]))
                                   for a, b in todo])
    for (a, b), s in zip(todo, scores_c1):
        all_pairs.append({"a": _cluster_frame(clusters[a]), "b": _cluster_frame(clusters[b]),
                          "label": float(s), "source": "c1_cluster"})
    n_c1 = len(todo)

    # ---- C2：簇×簇 同类型+不共享实体+时间≤30d（难负例新带，Jev 打分） ----
    c2_rows = db.execute("""
        SELECT k1.cluster_id c1, k2.cluster_id c2 FROM event_cluster k1
        JOIN event_cluster k2 ON k1.cluster_id < k2.cluster_id
        WHERE k1.deleted_at IS NULL AND k2.deleted_at IS NULL
          AND k1.state NOT IN ('redirected','split','deleted')
          AND k2.state NOT IN ('redirected','split','deleted')
          AND k1.event_type = k2.event_type
          AND ABS(julianday(COALESCE(k1.event_time_lower,'9999'))
                - julianday(COALESCE(k2.event_time_lower,'9999'))) <= 30
          AND NOT EXISTS (SELECT 1 FROM semantic_relation x1
                          JOIN semantic_relation x2 ON x1.from_id=x2.from_id
                          WHERE x1.to_id=k1.cluster_id AND x2.to_id=k2.cluster_id
                            AND x1.relation='participates_in' AND x2.relation='participates_in'
                            AND x1.deleted_at IS NULL AND x2.deleted_at IS NULL)
        ORDER BY RANDOM() LIMIT 25000""").fetchall()
    todo2 = [(r["c1"], r["c2"]) for r in c2_rows
             if r["c1"] in clusters and r["c2"] in clusters]
    print(f"C2 难负带 {len(todo2)} → Jev 打分…", flush=True)
    scores_c2 = judge.judge_batch([(_cluster_frame(clusters[a]), _cluster_frame(clusters[b]))
                                   for a, b in todo2])
    for (a, b), s in zip(todo2, scores_c2):
        all_pairs.append({"a": _cluster_frame(clusters[a]), "b": _cluster_frame(clusters[b]),
                          "label": float(s), "source": "c2_hard"})
    n_c2 = len(todo2)

    # ---- C4：正对补充（簇内提及对/转载/attach，原通道） ----
    n_c4 = 0
    for cid, mids in members.items():
        if len(mids) < 2:
            continue
        take = mids if len(mids) <= 5 else rng.sample(mids, 5)
        for i in range(len(take)):
            for j in range(i + 1, min(i + 4, len(take))):
                a, b = mentions.get(take[i]), mentions.get(take[j])
                if a and b and a["frame_text"] != b["frame_text"]:
                    all_pairs.append({"a": _mention_frame(a), "b": _mention_frame(b),
                                      "label": 1.0, "source": "c4_intra"})
                    n_c4 += 1
    for r in db.execute("""
            SELECT m1.mention_id a1, m2.mention_id a2 FROM event_mention m1
            JOIN event_mention m2 ON m1.mention_id < m2.mention_id
            JOIN document_version d1 ON d1.document_version_id=m1.document_version_id
            JOIN document_version d2 ON d2.document_version_id=m2.document_version_id
            JOIN source_lineage s1 ON s1.document_version_id=d1.document_version_id
            JOIN source_lineage s2 ON s2.document_version_id=d2.document_version_id
            WHERE s1.lineage_group=s2.lineage_group AND m1.status='valid' AND m2.status='valid'
              AND m1.frame_text != m2.frame_text LIMIT 2500"""):
        a, b = mentions.get(r["a1"]), mentions.get(r["a2"])
        if a and b:
            all_pairs.append({"a": _mention_frame(a), "b": _mention_frame(b),
                              "label": 1.0, "source": "c4_lineage"})
            n_c4 += 1

    # ---- C5：Max 合成正对（同一事件的另一种媒体报道写法） ----
    # 正对稀缺的根因：L3 已把库里高置信同事件簇合并收割。合成改写是质量最高的
    # 正对扩产（Max 按事件框架生成"另一家媒体对同一事件的报道句"，保持主体/
    # 对象/时间/动作，改变措辞与详略）——直击 v2 学生的 pos 层短板（措辞多样性）。
    import requests as _rq
    import os as _os
    _KEY = _os.environ.get("OR_KEY")
    pos_seed = [p for p in all_pairs if p["label"] >= 0.8][:8000]
    rng.shuffle(pos_seed)
    synth_cap = min(3000, len(pos_seed))
    n_c5 = 0
    print(f"C5 合成正对 {synth_cap} 条 → Max 改写…", flush=True)
    B5 = 8
    t5 = time.time()
    for bi in range(0, synth_cap, B5):
        chunk = pos_seed[bi:bi + B5]
        prompt_lines = [f"【事件{j+1}】{frame_text(x['a'])[:180]}" for j, x in enumerate(chunk)]
        got = {}
        for attempt in range(3):
            try:
                r = _rq.post("https://openrouter.ai/api/v1/chat/completions", timeout=150,
                             headers={"Authorization": f"Bearer {_KEY}"},
                             json={"model": MAX_MODEL, "temperature": 0.7, "max_tokens": 3500,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content":
                                     "对下面每个事件，写一句另一家媒体对同一事件的报道（保持主体、"
                                     "对象、时间、动作完全一致，改变措辞与详略，中文，不要编造新信息）。"
                                     '只输出JSON数组，每项形如 {"i": 编号, "text": "改写句"}：\n\n'
                                     + "\n".join(prompt_lines)}]})
                r.raise_for_status()
                c = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\[.*\]", c, re.S)
                if not m:
                    raise ValueError("no json")
                arr = json.loads(m.group())
                got = {int(x["i"]) - 1: x["text"] for x in arr if isinstance(x.get("text"), str)
                       and len(x.get("text", "")) > 10}
                break
            except Exception:
                time.sleep(3)
        for j, x in enumerate(chunk):
            if j in got:
                all_pairs.append({"a": x["a"],
                                  "b": {"frame": got[j][:160], "type": x["a"].get("type"),
                                        "time": x["a"].get("time"), "evidence": []},
                                  "label": 1.0, "source": "c5_synth"})
                n_c5 += 1
        if (bi // B5 + 1) % 40 == 0:
            print(f"  合成 {bi+B5}/{synth_cap} ({time.time()-t5:.0f}s)", flush=True)

    # ---- 去重 ----
    seen, uniq = set(), []
    for p in all_pairs:
        k = (frame_text(p["a"])[:80], frame_text(p["b"])[:80])
        rk = (k[1], k[0])
        if k in seen or rk in seen:
            continue
        seen.add(k)
        uniq.append(p)
    all_pairs = uniq
    print(f"去重后 {len(all_pairs)} 对（C1={n_c1} C2={n_c2} C3={n_c3} C4={n_c4}）", flush=True)

    # ---- Max 仲裁灰带（Jev 0.3-0.7）----
    n_arb = n_flip = 0
    if arbitrate:
        gray = [i for i, p in enumerate(all_pairs) if 0.3 <= p["label"] <= 0.7]
        print(f"灰带 {len(gray)} 对 → qwen3.8-max-0902 仲裁…", flush=True)
        # 只仲裁抽样上限 3 万对（成本与时间控制；其余保留 Jev 软标签）
        CAP = 30000
        arb_idx = gray if len(gray) <= CAP else rng.sample(gray, CAP)
        sub = [{"a": all_pairs[i]["a"], "b": all_pairs[i]["b"]} for i in arb_idx]
        max_scores = max_arbitrate(sub, None)
        for i, ms in max_scores.items():
            idx = arb_idx[i]
            jev_dir = all_pairs[idx]["label"] >= 0.5
            max_dir = ms >= 0.5
            n_arb += 1
            if max_dir != jev_dir:
                # 用户指定 Max 为准确打标器：分歧时以 Max 为准（硬方向），保留 Jev 软值于 source
                all_pairs[idx]["label"] = 1.0 if max_dir else 0.0
                all_pairs[idx]["source"] += "_maxflip"
                n_flip += 1
            else:
                # 一致 → 强化标签（提纯灰带为高置信）
                all_pairs[idx]["label"] = (max(all_pairs[idx]["label"], 0.75) if jev_dir
                                           else min(all_pairs[idx]["label"], 0.25))
        print(f"仲裁完成：{n_arb} 对，Max 推翻 Jev {n_flip} 对（{n_flip/max(n_arb,1):.1%}）",
              flush=True)

    # ---- 下采样负对到正对的 6 倍 + 总量控制 ----
    pos = [p for p in all_pairs if p["label"] >= 0.5]
    neg = [p for p in all_pairs if p["label"] < 0.5]
    rng.shuffle(neg)
    neg = neg[: min(len(neg), len(pos) * 6, max_pairs - len(pos))]
    final = pos + neg
    rng.shuffle(final)

    # ---- 时间切分 8:1:1 ----
    def sort_key(p):
        return (p["a"].get("time") or "?").split("~")[0]
    final.sort(key=sort_key)
    n = len(final)
    splits = {"train": final[:int(n * .8)], "val": final[int(n * .8):int(n * .9)],
              "test": final[int(n * .9):]}

    l1_out, l2_out = [], []
    for split, items in splits.items():
        neg_pool = [p for p in items if p["label"] < 0.5]
        for p in items:
            l2_out.append({"a": frame_text(p["a"]), "b": frame_text(p["b"]),
                           "label": round(p["label"], 4), "source": p["source"], "split": split})
            if p["label"] >= 0.75 and split == "train":
                negs = rng.sample(neg_pool, min(4, len(neg_pool))) if neg_pool else []
                l1_out.append({"anchor": frame_text(p["a"]), "positive": frame_text(p["b"]),
                               "negatives": [frame_text(x["b"]) for x in negs]})

    (DATA_DIR / "l1_pairs.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in l1_out), encoding="utf-8")
    (DATA_DIR / "l2_pairs.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in l2_out), encoding="utf-8")
    from collections import Counter
    stats = {"version": "v3", "seed": SEED, "total": n, "pos": len(pos), "neg": len(neg),
             "arbitrated": n_arb, "max_flips": n_flip,
             "channels": {"c1_cluster": n_c1, "c2_hard": n_c2, "c3_psame": n_c3, "c4_pos": n_c4, "c5_synth": n_c5},
             "pos_sources": dict(Counter(p["source"] for p in pos)),
             "splits": {k: {"n": len(v), "pos": sum(1 for x in v if x["label"] >= 0.5)}
                        for k, v in splits.items()},
             "l1_rows": len(l1_out), "l2_rows": len(l2_out)}
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    db.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=200000)
    ap.add_argument("--no-arbitrate", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build(a.max_pairs, not a.no_arbitrate), ensure_ascii=False, indent=1))
