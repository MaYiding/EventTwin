# -*- coding: utf-8 -*-
"""统一评测器：任意判别器 × 金标 benchmark → AUROC / 最优阈值 acc / ECE / 分层报告。

用法：
  python3 -m ml.evaluator jev                     # Jev（复用预标缓存，秒级）
  python3 -m ml.evaluator embedding               # 现役 embedding-8b（通用表示基线）
  python3 -m ml.evaluator reranker                # 未微调 bge-reranker（零训练基线，需GPU）
  python3 -m ml.evaluator cascade --band 0.35     # 级联（本地+Jev 终审）
  python3 -m ml.evaluator reranker --model ml/models/ckpt_x   # 训后学生
金标以 human 为准（复核后）；未复核时用 jev>=0.5 临时标签并在报告标注。
报告落 ml/benchmark/report_<name>.json + 控制台 markdown 表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BM_DIR = Path(__file__).parent / "benchmark"
REPORT_DIR = BM_DIR


def load_benchmark():
    for name in ("bm_v1_human.json", "bm_v1_dual.json", "bm_v1_partial.json", "bm_v1.json"):
        p = BM_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("先运行 python3 -m ml.benchmark build")


def _auroc(labels, scores):
    pos = sorted(s for s, y in zip(scores, labels) if y == 1)
    neg = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not pos or not neg:
        return None
    # 秩法 AUC（平均秩处理并列）
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rp = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0)


def _best_acc(labels, scores):
    cands = sorted(set(scores))
    best = (0.0, 0.5)
    for t in cands:
        acc = sum(1 for y, s in zip(labels, scores) if (s >= t) == (y == 1)) / len(labels)
        if acc > best[0]:
            best = (acc, t)
    return best


def _ece(labels, scores, bins=10):
    """期望校准误差（按分数分 10 桶，|平均分-正例率| 加权）。"""
    import collections
    bucket = collections.defaultdict(list)
    for y, s in zip(labels, scores):
        bucket[min(int(s * bins), bins - 1)].append((y, s))
    n = len(labels)
    return sum(len(v) / n * abs(sum(y for y, _ in v) / len(v) - sum(s for _, s in v) / len(v))
               for v in bucket.values())


def evaluate(judge_name: str, model: str | None = None, band: float | None = None) -> dict:
    from .judges import make_judge
    bm = load_benchmark()
    human_cover = sum(1 for p in bm["pairs"] if p["human"] is not None)
    if human_cover >= len(bm["pairs"]) * 0.8:
        label_src = "human"
    else:
        label_src = "jev(未人工复核)"

    judge = make_judge(judge_name, **({"model": model} if model else {}),
                       **({"band": band} if band is not None and judge_name == "cascade" else {}))
    pairs = [(p["a"], p["b"]) for p in bm["pairs"]]
    print(f"评测 {judge.name} × {len(pairs)} 对（金标={label_src}）…")
    scores = judge.judge_batch(pairs)

    labels = [int(p["human"]) if p["human"] is not None else int(p["jev_score"] >= 0.5)
              for p in bm["pairs"]]
    layers = [p["layer"] for p in bm["pairs"]]

    def metrics(ids):
        ls = [labels[i] for i in ids]
        ss = [scores[i] for i in ids]
        if not ls:
            return None
        acc, thr = _best_acc(ls, ss)
        auroc = _auroc(ls, ss)
        return {"n": len(ids), "auroc": round(auroc, 4) if auroc is not None else None,
                "acc@最优阈": round(acc, 4), "阈值": round(thr, 3),
                "ece": round(_ece(ls, ss), 4)}

    # 可靠性分桶（reliability diagram 数据）：10 桶 置信 vs 实际正率——
    # 校准诊断（过自信/欠自信发生在哪个分数带），支撑温度/阈值迭代
    import collections
    bucket = collections.defaultdict(lambda: [0, 0.0])
    for y, s in zip(labels, scores):
        b = bucket[min(int(s * 10), 9)]
        b[0] += 1
        b[1] += y
    reliability = [{"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": v[0],
                    "conf": round((i + 0.5) / 10, 2), "acc": round(v[1] / v[0], 4)}
                   for i, v in sorted(bucket.items()) if v[0]]

    report = {
        "judge": judge.name, "model": model, "label_source": label_src,
        "overall": metrics(range(len(pairs))),
        "by_layer": {layer: metrics([i for i, l in enumerate(layers) if l == layer])
                     for layer in sorted(set(layers))},
        "reliability": reliability,
        "vs_jev_agreement": None,
    }
    # 与教师的一致率（学生模型的关键指标；fidelity）
    if judge_name != "jev":
        jev = [p["jev_score"] for p in bm["pairs"]]
        agree = sum(1 for j, s in zip(jev, scores) if (j >= 0.5) == (s >= 0.5)) / len(pairs)
        report["vs_jev_agreement"] = round(agree, 4)
    if judge_name == "cascade" and getattr(judge, "last_escalation_rate", None) is not None:
        report["escalation_rate"] = round(judge.last_escalation_rate, 4)

    out = REPORT_DIR / f"report_{judge.name}{'_' + model.replace('/', '_') if model else ''}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n| 层 | n | AUROC | acc@最优阈 | 阈值 | ECE |")
    print("|---|---|---|---|---|---|")
    rows = [("OVERALL", report["overall"])] + list(report["by_layer"].items())
    for name, m in rows:
        if m:
            print(f"| {name} | {m['n']} | {m['auroc']} | {m['acc@最优阈']} | {m['阈值']} | {m['ece']} |")
    if report["vs_jev_agreement"] is not None:
        print(f"\n与 Jev 硬判定一致率: {report['vs_jev_agreement']}")
    if "escalation_rate" in report:
        print(f"灰带升级率（API 调用占比）: {report['escalation_rate']}")
    print(f"\n报告已存 {out}")
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "jev"
    model = None
    band = None
    args = sys.argv[2:]
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--band" in args:
        band = float(args[args.index("--band") + 1])
    evaluate(name, model, band)
