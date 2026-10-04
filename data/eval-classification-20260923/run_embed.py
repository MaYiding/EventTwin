#!/usr/bin/env python3
"""Embedding 分类器：文本向量 vs 180 个叶子描述向量，余弦 argmax。"""
import json
import time

from common import ROOT, load_taxonomy, load_testset, leaf_desc, embed, cosine

TEXT_CAP = 1500  # 字符


def main():
    tax = load_taxonomy()
    items = load_testset()
    docs = [leaf_desc(tax, lf) for lf in tax["leaves"]]
    doc_vecs, u1 = embed(docs)
    print(f"leaf embeddings: {len(doc_vecs)} (usage={u1})", flush=True)

    out = []
    B = 16
    t0 = time.time()
    for s in range(0, len(items), B):
        batch = items[s:s + B]
        vecs, _ = embed([it["title"] + "\n" + it["content"][:TEXT_CAP]
                         for it in batch])
        for it, v in zip(batch, vecs):
            scores = [cosine(v, dv) for dv in doc_vecs]
            order = sorted(range(len(docs)), key=lambda i: -scores[i])
            out.append({"id": it["id"],
                        "top5": [docs[i].split("（")[0] for i in order[:5]],
                        "top5_score": [round(scores[i], 4) for i in order[:5]]})
        print(f"[{min(s+B, len(items))}/{len(items)}]", flush=True)
    for r in out:
        r["latency_total_s"] = round(time.time() - t0, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "embed.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("done")


if __name__ == "__main__":
    main()
