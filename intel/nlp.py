# -*- coding: utf-8 -*-
"""检索原语：中文分词（字符二元 + 英数词）、BM25、RRF 融合。

对齐铁律：BM25 与向量分数不直接相加，用 RRF 按排名融合；
时间过滤显式进行，不指望普通 Embedding 隐式学会时间。
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_LATIN = re.compile(r"[A-Za-z0-9]+")
_HAN = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中文按字符二元组（unigram+bigram），英数按词，统一小写。"""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    # 先按非中日韩/字母数字切段，保留中文段与英数段
    buf_han: list[str] = []
    for ch in text:
        if _HAN.match(ch):
            buf_han.append(ch)
        else:
            if buf_han:
                tokens.extend(_han_grams(buf_han))
                buf_han = []
    if buf_han:
        tokens.extend(_han_grams(buf_han))
    tokens.extend(_LATIN.findall(text))
    return tokens


def _han_grams(chars: list[str]) -> list[str]:
    out = []
    for i, ch in enumerate(chars):
        out.append(ch)  # unigram 保证召回
        if i + 1 < len(chars):
            out.append(ch + chars[i + 1])  # bigram 提升区分度
    return out


class BM25Index:
    """内存 BM25（k1=1.2, b=0.75）。索引可随时从 DB 文本重建（派生层铁律）。"""

    def __init__(self):
        self.doc_ids: list[str] = []
        self.doc_len: dict[str, int] = {}
        self.tf: dict[str, Counter] = {}
        self.df: Counter = Counter()
        self.n = 0
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        if doc_id in self.doc_len:  # 幂等：重建时覆盖
            return
        self.doc_ids.append(doc_id)
        self.doc_len[doc_id] = len(toks)
        c = Counter(toks)
        self.tf[doc_id] = c
        for t in c:
            self.df[t] += 1
        self.n += 1
        self.avgdl = sum(self.doc_len.values()) / max(self.n, 1)

    def score(self, query: str) -> list[tuple[str, float]]:
        q = set(tokenize(query))
        if not q or self.n == 0:
            return []
        k1, b = 1.2, 0.75
        out = []
        for doc_id in self.doc_ids:
            dl = self.doc_len[doc_id]
            if dl == 0:
                continue
            c = self.tf[doc_id]
            s = 0.0
            for t in q:
                if t not in c:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = c[t]
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                out.append((doc_id, s))
        out.sort(key=lambda x: -x[1])
        return out

    def top(self, query: str, k: int) -> list[str]:
        return [d for d, _ in self.score(query)[:k]]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：多路排名融合，不合并原始分数。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
