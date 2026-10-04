# -*- coding: utf-8 -*-
"""训练数据 → ms-swift 官方格式转换（L1 embedding InfoNCE / L2 reranker ranking）。

依据 L1L2L3-实施方案 §8.1/8.3：
- L1 主路 = Qwen3-Embedding-4B 全参 ZeRO-3（SWIFT：--task_type embedding
  --model_type qwen3_emb --loss_type infonce，跨卡负例 INFONCE_USE_BATCH=True +
  假负例 mask INFONCE_MASK_FAKE_NEGATIVE=True + drop_last）；
- L2 主路 = 自定义软标签 BCE 蒸馏（train_l2_reranker.py，直传 Jev 校准概率），
  SWIFT ranking 格式作对照组（硬标签排序式）。

产出 ml/data/swift/：
  l1_train.jsonl / l1_val.jsonl      {"query":…, "pos":[…], "neg":[…]}   （embedding InfoNCE）
  l2_train.jsonl / l2_val.jsonl      同构                                （reranker ranking 对照）
用法：python3 -m ml.swift_data
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path(__file__).parent / "data"
OUT = SRC / "swift"
SEED = 20260930


def convert_l1() -> dict:
    """l1_pairs（anchor/positive/negatives）→ SWIFT embedding 三元组。"""
    rows = [json.loads(x) for x in (SRC / "l1_pairs.jsonl").read_text(encoding="utf-8").splitlines() if x]
    # split 信息在 l2 文件里——l1 只由 train 正对构成，这里按同序重建 val（用 l2 val 的对）
    l2 = [json.loads(x) for x in (SRC / "l2_pairs.jsonl").read_text(encoding="utf-8").splitlines() if x]
    val_pos = [r for r in l2 if r["split"] == "val" and r["label"] >= 0.9]
    val_neg = [r for r in l2 if r["split"] == "val" and r["label"] <= 0.1]
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    def to_swift(anchor, pos, negs):
        return {"query": anchor, "pos": [pos], "neg": negs}

    train = [to_swift(r["anchor"], r["positive"], r["negatives"]) for r in rows if r["negatives"]]
    val = []
    for p in val_pos:
        negs = [n["b"] for n in rng.sample(val_neg, min(7, len(val_neg)))] if val_neg else []
        if negs:
            val.append(to_swift(p["a"], p["b"], negs))
    _dump(OUT / "l1_train.jsonl", train)
    _dump(OUT / "l1_val.jsonl", val)
    return {"l1_train": len(train), "l1_val": len(val)}


def convert_l2(max_neg_per_pos: int = 7) -> dict:
    """l2_pairs（软标签）→ SWIFT reranker ranking 对照格式（MAX_POS=1/MAX_NEG=7）。
    软标签值 >0.5 入 pos、<0.5 入 neg（ranking 式丢软标签信息，仅作对照）。
    """
    rows = [json.loads(x) for x in (SRC / "l2_pairs.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rng = random.Random(SEED)
    train, val = [], []
    for split, sink in (("train", train), ("val", val)):
        items = [r for r in rows if r["split"] == split]
        pos = [r for r in items if r["label"] > 0.5]
        neg = [r for r in items if r["label"] <= 0.5]
        for p in pos:
            negs = [n["b"] for n in rng.sample(neg, min(max_neg_per_pos, len(neg)))] if neg else []
            if negs:
                sink.append({"query": p["a"], "pos": [p["b"]], "neg": negs})
    _dump(OUT / "l2_train.jsonl", train)
    _dump(OUT / "l2_val.jsonl", val)
    return {"l2_train": len(train), "l2_val": len(val)}


def _dump(path: Path, rows: list) -> None:
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")


if __name__ == "__main__":
    stats = {}
    stats.update(convert_l1())
    stats.update(convert_l2())
    print(json.dumps(stats, ensure_ascii=False, indent=1))
