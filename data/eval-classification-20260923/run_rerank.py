#!/usr/bin/env python3
"""Rerank 分类器：query=标题+正文摘录，180 叶子分 4 块打分取全局 argmax。"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, load_taxonomy, load_testset, leaf_desc, rerank

Q_CONTENT = 220  # query 里正文摘录字符数（llama.cpp 每 pair ≤512 token 限制）
CHUNKS = 4


def run_one(args):
    item, docs = args
    query = item["title"] + "\n" + item["content"][:Q_CONTENT]
    t0 = time.time()
    all_scores = [0.0] * len(docs)
    try:
        step = (len(docs) + CHUNKS - 1) // CHUNKS
        for s in range(0, len(docs), step):
            sc = rerank(query, docs[s:s + step])
            for j, v in enumerate(sc):
                all_scores[s + j] = v
        order = sorted(range(len(docs)), key=lambda i: -all_scores[i])
        return {"id": item["id"],
                "top5": [docs[i].split("（")[0] for i in order[:5]],
                "top5_score": [round(all_scores[i], 4) for i in order[:5]],
                "latency_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"id": item["id"], "error": str(e)[:300],
                "latency_s": round(time.time() - t0, 2)}


def main():
    tax = load_taxonomy()
    items = load_testset()
    docs = [leaf_desc(tax, lf) for lf in tax["leaves"]]
    print(f"{len(items)} texts, {len(docs)} leaves, {CHUNKS} chunks", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, res in enumerate(ex.map(run_one,
                                       [(it, docs) for it in items])):
            out.append(res)
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(items)}] {res.get('top5',[None])[0]} "
                      f"{res['latency_s']}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rerank.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"done: {sum(1 for r in out if 'top5' in r)}/{len(out)} ok")


if __name__ == "__main__":
    main()
