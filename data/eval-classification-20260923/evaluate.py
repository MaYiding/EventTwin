#!/usr/bin/env python3
"""v3 汇总：5 模型 × 3 任务，基于人工修正后的 262 条精确测试集重打分。

标注修正只改参考答案、不改模型输入输出，因此对既有四模型重打分等价于重跑；
Laya 为全新推理。指标在 v2 基础上扩充：宏F1(L3)、实体分组检出、事件伪造严重度
召回、事件分维对比、Jev 置信度分带。
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from common import ROOT, load_taxonomy, load_testset, auc, best_acc

TRUE_CLAIMS = {"c01", "c12", "c13", "c21", "c22", "c30", "c31", "c35"}
FORGED = {"c02": "issuer", "c03": "issuer", "c04": "issuer+receiver",
          "c05": "receiver", "c06": "receiver", "c07": "relation",
          "c08": "relation", "c09": "relation", "c10": "relation",
          "c11": "time", "c14": "time", "c15": "time", "c16": "time",
          "c17": "time", "c18": "issuer", "c19": "issuer", "c20": "relation",
          "c23": "time", "c24": "time", "c25": "time", "c26": "relation",
          "c27": "relation", "c28": "issuer", "c29": "issuer",
          "c32": "issuer", "c33": "time", "c34": "relation", "c36": "time",
          "c37": "time", "c38": "time", "c39": "receiver", "c40": "relation"}
SEVERITY = {  # 伪造严重度
    "fine": ["c07", "c14", "c15", "c23", "c27", "c36", "c37"],   # 数值/日期微调
    "mid": ["c02", "c03", "c05", "c09", "c11", "c18", "c19", "c26",
            "c28", "c29", "c33", "c40", "c06", "c25", "c16", "c38", "c39"],
    "coarse": ["c04", "c08", "c10", "c17", "c20", "c24", "c32", "c34"]}


def cls_metrics():
    tax = load_taxonomy()
    items = {it["id"]: it for it in load_testset()}
    id2path = {lf["id"]: "/".join(lf["path"]) for lf in tax["leaves"]}
    out = {}
    for name in ("jev", "laya", "rerank", "qwen8b", "embed"):
        f = ROOT / "results" / f"{name}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        c = defaultdict(int)
        hf1, lats, cost = [], [], []
        l4_ok = l4_n = 0
        l3_tp, l3_fp, l3_fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for r in rows:
            if r["id"] not in items:  # 修正层剔除的样本
                continue
            gold = items[r["id"]]["gold_path"]
            gpath = "/".join(gold)
            if name == "jev":
                if "choice" not in r or r["choice"] not in id2path:
                    c["invalid"] += 1
                    continue
                ppath = id2path[r["choice"]]
                top = [id2path.get(k, "?") for k in r.get("top3", [])]
                lats.append(r["latency_s"])
                if r.get("cost"):
                    cost.append(r["cost"])
            elif name in ("qwen8b", "laya"):
                ppath = "/".join(r.get("pred_path", []))
                top = [ppath]
                lats.append(r["latency_s"])
                if "error" in r:
                    c["invalid"] += 1
            else:
                if "top5" not in r:
                    c["invalid"] += 1
                    continue
                ppath = r["top5"][0].split("——")[0]
                top = [t.split("——")[0] for t in r["top5"]]
                if r.get("latency_s"):
                    lats.append(r["latency_s"])
            pred = ppath.split("/")
            c["n"] += 1
            if ppath == gpath:
                c["leaf1"] += 1
            if gpath in top[:3]:
                c["leaf3"] += 1
            for k in range(3):
                if len(pred) > k and pred[k] == gold[k]:
                    c[f"L{k+1}"] += 1
            if len(gold) == 4:
                l4_n += 1
                l4_ok += (len(pred) == 4 and pred[3] == gold[3])
            gl3, pl3 = gold[2], (pred[2] if len(pred) > 2 else "∅")
            if gl3 == pl3:
                l3_tp[gl3] += 1
            else:
                l3_fn[gl3] += 1
                l3_fp[pl3] += 1
            gp = {"/".join(gold[:i + 1]) for i in range(len(gold))}
            pp = {"/".join(pred[:i + 1]) for i in range(len(pred))}
            inter = len(gp & pp)
            p = inter / len(pp) if pp else 0
            rc = inter / len(gp) if gp else 0
            hf1.append(2 * p * rc / (p + rc) if p + rc else 0)
        m = {k: round(v / c["n"], 4) for k, v in c.items()
             if k not in ("n", "invalid") and c["n"]}
        m["n"] = c["n"]
        m["invalid"] = c["invalid"]
        m["L4_cond"] = round(l4_ok / l4_n, 4) if l4_n else None
        m["hF1"] = round(sum(hf1) / len(hf1), 4) if hf1 else 0
        # 宏 F1（L3 事件类型，类平衡）
        f1s = []
        for k in set(l3_tp) | set(l3_fn):
            tp = l3_tp[k]
            prec = tp / (tp + l3_fp[k]) if tp + l3_fp[k] else 0
            rec = tp / (tp + l3_fn[k]) if tp + l3_fn[k] else 0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        m["macroF1_L3"] = round(sum(f1s) / len(f1s), 4) if f1s else None
        if lats:
            m["latency_mean_s"] = round(st.mean(lats), 2)
        if cost:
            m["cost_total_usd"] = round(sum(cost), 4)
        out[name] = m
    return out


def ev_row(scores):
    labels = [1 if s["label"] == 1 else 0 for s in scores]
    vals = [0.5 if s["score"] is None else s["score"] for s in scores]
    a = auc(labels, vals)
    return {"acc@0.5": round(sum((v >= .5) == (l == 1) for v, l in
                                 zip(vals, labels)) / len(labels), 4),
            "best_acc": round(best_acc(labels, vals), 4),
            "AUC": round(a, 4) if a is not None else None}


def ent_ev_metrics():
    f = ROOT / "results" / "extra_entity_event.json"
    if not f.exists():
        return {"entity": {}, "event": {}}
    x = json.loads(f.read_text())
    out = {"entity": {}, "event": {}}

    def group_acc(rows, key="group"):
        g = defaultdict(lambda: [0, 0])
        for e in rows:
            lab = 1 if e[key] == "A" else 0
            v = 0.5 if e["score"] is None else e["score"]
            g[e[key]][1] += 1
            g[e[key]][0] += ((v >= .5) == (lab == 1))
        return {k: f"{a}/{b}" for k, (a, b) in sorted(g.items())}

    for key, model in (("entity_jev", "jev"), ("entity_laya", "laya"),
                       ("entity_8b", "qwen8b"), ("entity_embed", "embed"),
                       ("entity_rerank", "rerank")):
        if key in x:
            d = ev_row([{"label": 1 if e["group"] == "A" else 0,
                         "score": e["score"]} for e in x[key]])
            d["per_group@0.5"] = group_acc(x[key])
            out["entity"][model] = d

    for key, model in (("event_jev", "jev"), ("event_laya", "laya"),
                       ("event_8b", "qwen8b"), ("event_embed", "embed"),
                       ("event_rerank", "rerank")):
        if key in x:
            d = ev_row([{"label": 1 if e["id"] in TRUE_CLAIMS else 0,
                         "score": e.get("overall", e.get("score"))}
                        for e in x[key]])
            # 伪造严重度召回（伪造声明被判否的比例）
            sev = {}
            for s, ids in SEVERITY.items():
                rows = [e for e in x[key] if e["id"] in ids]
                rows = [e for e in rows if e.get("overall") is not None]
                if rows:
                    sev[s] = f"{sum(1 for e in rows if e['overall'] < .5)}/{len(rows)}"
            if sev:
                d["forgery_recall@0.5"] = sev
            out["event"][model] = d

    # 事件分维（有分维输出的模型）
    for key, model in (("event_jev", "jev"), ("event_laya", "laya"),
                       ("event_8b", "qwen8b")):
        if key not in x:
            continue
        dim_ok, dim_n = defaultdict(int), defaultdict(int)
        for e in x[key]:
            fg = FORGED.get(e["id"], "")
            for d in ("issuer", "receiver", "relation", "time"):
                if e.get(d) is None:
                    continue
                gold = 0 if d in fg.split("+") else 1
                dim_n[d] += 1
                if (e[d] >= .5) == (gold == 1):
                    dim_ok[d] += 1
        out["event"][model]["per_dim_acc@0.5"] = {
            d: f"{dim_ok[d]}/{dim_n[d]}" for d in dim_n}
    return out


def jev_bands():
    """Jev 分类置信度分带（修正标注后）。"""
    tax = load_taxonomy()
    items = {it["id"]: it for it in load_testset()}
    id2path = {lf["id"]: "/".join(lf["path"]) for lf in tax["leaves"]}
    rows = [r for r in json.loads((ROOT / "results" / "jev.json").read_text())
            if r["id"] in items]
    bands = {}
    for t in (0.5, 0.6, 0.7, 0.8):
        n_auto = sum(1 for r in rows if r.get("confidence", 0) >= t)
        acc = sum(1 for r in rows if r.get("confidence", 0) >= t
                  and id2path.get(r["choice"]) ==
                  "/".join(items[r["id"]]["gold_path"]))
        bands[f">={t}"] = f"{acc}/{n_auto}" + (f"（P={acc/max(n_auto,1):.3f}）"
                                               if n_auto else "")
    return bands


def main():
    m = {"classification": cls_metrics(), **ent_ev_metrics(),
         "jev_conf_bands": jev_bands()}
    (ROOT / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
