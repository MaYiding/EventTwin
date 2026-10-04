#!/usr/bin/env python3
"""对比实验：qwen3-reranker-0.6b vs TypeSafe noul 的"实体是否在文中"判别。

两个口径：
- full  : 标题+正文全量（对标 TypeSafe 那次请求）
- trunc : 生产口径，文档截断到 rerank_max_doc_chars=1200 字符
query 一律用实体名称本身（与 TypeSafe 的 query_entity.名称 一致）。
"""

import json
import time
import urllib.request

BASE = "http://100.92.114.89:8083/v1/rerank"
MODEL = "qwen3-reranker-0.6b"
TRUNC_CHARS = 1200  # 与 Code/config.json 的 rerank_max_doc_chars 一致

TITLE = "Apple settlement offers eligible iPhone owners up to $95. Here is how to file a claim."

BODY = """The claim window is now open for an Apple settlement announced earlier this year, giving certain iPhone users the chance to get up to $95.

Apple agreed to pay $250 million to settle claims that it misled customers about Siri's Apple Intelligence features. Customers who were part of the class action lawsuit said they purchased iPhones expecting certain Siri features they allegedly did not receive.

"The $250 million settlement fund represents the largest false advertising settlement in history," Ryan Clarkson, co-lead counsel on the class-action lawsuit and founder and managing partner of Clarkson Law Firm, said in an email to CBS News.

Apple declined to comment. The company has previously denied the allegations. According to the settlement agreement, the settlement shall "not be construed in any fashion as an admission of liability or wrongdoing by Apple."

Here's what to know about eligibility and how to file a claim.

How do I file a claim?

iPhone users can visit SmartphoneAISettlement.com to file a claim, according to a spokesperson for Clarkson Law Firm, a public interest law firm.

The person must input basic personal information, including their name, address and phone number. They must also enter their iPhone's serial number, which can be found in the phone's Settings section.

Once they finish inputting that information, they can select the payment method. The options are PayPal, Venmo, direct deposit or check.

As SmartphoneAISettlement.com points out, consumers are limited to "one cash payment per eligible device." If they purchased another iPhone that meets the eligibility requirements, they must complete a separate claim form.

How do I know if I am eligible?

If you bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 10, 2024, and March 29, 2025, you're eligible to file a claim.

According to SmartphoneAISettlement.com, the person filing a claim must be the person who originally purchased the device. The phone must also be for personal or professional use, not for resale.

If you're not sure if you're eligible, you can call 1-888-988-8945 or write to this address for information: Landsheft, et al. v. Apple Inc.; Settlement Administrator; P.O. Box 301132; Los Angeles, CA 90030-1132.

How long do I have to file a claim?

The claims window is open until Dec. 21, 2026.

How much money will I receive?

Customers who submit a valid claim form will initially get $25 per eligible device. However, that amount may be increased up to $95 or decreased depending on the "total number of valid claims submitted and other factors," the claims website says.

When will I get the payment?

A district court in San Jose, California, will hold a hearing on Feb. 24, 2027, to decide whether to approve the settlement. If there are any appeals, it could delay the payment process.

"If there is no appeal, your settlement benefit will be processed promptly," according to the claim website."""

DOC_FULL = TITLE + "\n\n" + BODY
DOC_TRUNC = DOC_FULL[:TRUNC_CHARS]

# (key, query, 组, TypeSafe noul 实测值)
ENTITIES = [
    ("apple",               "Apple",                       "A", 0.99),
    ("apple_inc",           "Apple Inc.",                  "A", 0.98),
    ("iphone",              "iPhone",                      "A", 0.99),
    ("siri",                "Siri",                        "A", 0.99),
    ("ryan_clarkson",       "Ryan Clarkson",               "A", 0.99),
    ("clarkson_law_firm",   "Clarkson Law Firm",           "A", 0.95),
    ("paypal",              "PayPal",                      "A", 0.98),
    ("landsheft",           "Landsheft",                   "A", 0.92),
    ("samsung",             "Samsung",                     "B", 0.02),
    ("google",              "Google",                      "B", 0.02),
    ("elon_musk",           "Elon Musk",                   "B", 0.02),
    ("microsoft",           "Microsoft",                   "B", 0.02),
    ("amazon",              "Amazon",                      "B", 0.02),
    ("ipad",                "iPad",                        "C", 0.02),
    ("iphone_14_pro",       "iPhone 14 Pro",               "C", 0.03),
    ("android",             "Android",                     "C", 0.03),
    ("ftc",                 "Federal Trade Commission",    "C", 0.03),
    ("clarson_law_firm",    "Clarson Law Firm",            "D", 0.20),
    ("appel_inc",           "Appel Inc.",                  "D", 0.30),
    ("ryan_clark",          "Ryan Clark",                  "D", 0.26),
    ("clarkson_university", "Clarkson University",         "D", 0.02),
    ("venmon",              "Venmon",                      "D", 0.49),
]


def rerank(query: str, documents: list[str]) -> list[float]:
    req = urllib.request.Request(
        BASE,
        data=json.dumps({
            "model": MODEL, "query": query,
            "documents": documents, "top_n": len(documents),
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read())
    scores = [0.0] * len(documents)
    for r in body.get("results", []):
        scores[r.get("index", 0)] = float(r.get("relevance_score", 0.0))
    return scores


# 全文物理上超服务端 batch（512 token）限制，改为 ~1100 字符分块覆盖全文，
# 取实体对各块的最大分 —— RAG 中 chunk 级 rerank 的标准用法
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 50
_chunks = []
_start = 0
while _start < len(DOC_FULL):
    _chunks.append(DOC_FULL[_start:_start + CHUNK_SIZE])
    _start += CHUNK_SIZE - CHUNK_OVERLAP
DOC_CHUNKS = _chunks


def main() -> None:
    print(f"全文 {len(DOC_FULL)} 字符 -> {len(DOC_CHUNKS)} 块 "
          f"({[len(c) for c in DOC_CHUNKS]})", flush=True)
    rows = []
    for key, query, group, noul in ENTITIES:
        chunk_scores = rerank(query, DOC_CHUNKS)
        s_trunc = rerank(query, [DOC_TRUNC])[0]
        s_full = max(chunk_scores)
        rows.append({
            "key": key, "query": query, "group": group,
            "noul": noul, "rerank_full": round(s_full, 4),
            "rerank_trunc": round(s_trunc, 4),
            "chunk_scores": [round(s, 4) for s in chunk_scores],
        })
        print(f"[{group}] {query:<26} noul={noul:.2f}  "
              f"full_max={s_full:.4f}  trunc={s_trunc:.4f}", flush=True)
        time.sleep(0.15)

    with open("/tmp/rerank_vs_typesafe.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 分组极差对比
    print("\n=== 分组 [min, max] ===")
    for g, name in [("A", "在文中"), ("B", "无关"), ("C", "相近不在"), ("D", "伪造/近似")]:
        sub = [r for r in rows if r["group"] == g]
        for field in ("noul", "rerank_full", "rerank_trunc"):
            vals = [r[field] for r in sub]
            print(f"{name} {field:<12} [{min(vals):.2f}, {max(vals):.2f}]")
        print()

    # Spearman（noul vs rerank_full / rerank_trunc）
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r

    def spearman(a, b):
        ra, rb = rank(a), rank(b)
        n = len(a)
        ma, mb = sum(ra) / n, sum(rb) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        da = (sum((x - ma) ** 2 for x in ra)) ** 0.5
        db = (sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / (da * db)

    noul = [r["noul"] for r in rows]
    print(f"Spearman(noul, rerank_full)  = {spearman(noul, [r['rerank_full'] for r in rows]):.3f}")
    print(f"Spearman(noul, rerank_trunc) = {spearman(noul, [r['rerank_trunc'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
