#!/usr/bin/env python3
"""公共库：数据加载、HTTP、指标。"""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
CHAT_URL = "http://100.92.114.89:8081/v1/chat/completions"
EMBED_URL = "http://100.92.114.89:8082/v1/embeddings"
RERANK_URL = "http://100.92.114.89:8083/v1/rerank"
OR_DECISIONS = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13-20260917"


def load_taxonomy():
    return json.loads((ROOT / "taxonomy.json").read_text())


def load_testset():
    return json.loads((ROOT / "testset.json").read_text())["items"]


def leaf_desc(tax, leaf):
    """叶子给模型的描述：路径 + L2/L3/L4 语义说明（v2 体系三段全带）。"""
    path = leaf["path"]
    parts = []
    if path[1] in tax.get("l2_desc", {}):
        parts.append(tax["l2_desc"][path[1]])
    if path[2] in tax.get("l3_desc", {}):
        parts.append(tax["l3_desc"][path[2]])
    if len(path) == 4:
        parts.append(tax["l4_tree"].get(path[2], {}).get(path[3], ""))
    return "/".join(path) + "——" + "；".join(x for x in parts if x)


def http_json(url, payload, timeout=180, headers=None, retries=3):
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", **(headers or {})},
                method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} 失败: {last}")


def or_decisions(payload):
    key = os.environ["OR_KEY"]
    return http_json(OR_DECISIONS, payload, timeout=240,
                     headers={"Authorization": f"Bearer {key}"})


def chat(prompt, max_tokens=48, temperature=0.0):
    body = {"model": "qwen3-8b", "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt + " /no_think"}]}
    r = http_json(CHAT_URL, body, timeout=300)
    return r["choices"][0]["message"]["content"]


def embed(texts):
    r = http_json(EMBED_URL, {"model": "qwen3-embedding-4b", "input": texts},
                 timeout=300)
    emap = {i: d["embedding"] for i, d in enumerate(r["data"])}
    return [emap[i] for i in range(len(texts))], r.get("usage", {})


def rerank(query, documents, top_n=None):
    r = http_json(RERANK_URL, {"model": "qwen3-reranker-0.6b", "query": query,
                               "documents": documents,
                               "top_n": top_n or len(documents)}, timeout=120)
    scores = [0.0] * len(documents)
    for x in r.get("results", []):
        scores[x.get("index", 0)] = float(x.get("relevance_score", 0.0))
    return scores


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def auc(labels, scores):
    """秩法 AUC（平均秩处理并列）。"""
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
    pos = [r for r, y in zip(ranks, labels) if y == 1]
    neg = [r for r, y in zip(ranks, labels) if y == 0]
    if not pos or not neg:
        return None
    return (sum(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_acc(labels, scores):
    """最优阈值准确率。"""
    cands = sorted(set(scores))
    best = 0.0
    for t in cands:
        acc = sum(1 for y, s in zip(labels, scores) if (s >= t) == (y == 1)) / len(labels)
        best = max(best, acc)
    return best
