# EventTwin

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-EventTwin%20v2.0-yellow)](https://huggingface.co/MaYiding/EventTwin)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EventTwin--Data-orange)](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20Transformers-red)](https://pytorch.org)
[![Language](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](#-中文文档)

**EventTwin** (latest: **v2.0**) is a 568M Chinese cross-encoder that answers one precise question:

> *Do these two event descriptions refer to the **same real-world event**?*

It outputs a **temperature-calibrated probability** (not a similarity score) — so its numbers can be
used directly as confidence for routing, gating, and cascade pipelines. The repo ships a **six-version family across two base models** (main = v2.0 on
Qwen3-Reranker-4B; tags `v1.4`/`v1.3`/`v1.2`/`v1.1`/`v1.0` on a 568M encoder). **v2.0** is
the biggest leap yet: **overall AUROC 0.963** (v1.x best 0.909), **gray-zone AUROC 0.974**
(the long-standing 0.866 ceiling closed), **ECE 0.105**, agreement-with-teacher **0.918**,
and the most decisive band behavior (10.7% escalation, 72.7% auto-merge coverage).
**v1.2** remains the zero-false-merge gate (0% at ≥0.9); **v1.0** the lightweight standalone.
See [the version comparison](#versions) — all on a 1,000-pair dual-teacher gold, 2-5 ms/pair
(568M) or ~20 ms/pair (4B) on a single GPU, with full benchmark protocol, training data, and
a training archive (including failed iterations and extracted data laws).

## Quick start

```bash
pip install torch transformers
```

```python
# v2.0（main，4B decoder）——三条协议红线：200字截断 / 左padding取末位 / 除温度T
from inference_v2 import load, same_event_prob   # 本仓库 inference_v2.py
tok, model, T, yes_id, no_id = load("MaYiding/EventTwin")
same_event_prob("小米YU7正式上市，售价25.35万元起，共推出三款车型",
                "小米发布YU7系列 售价25.35万起")          # ≈ 0.93

# v1.x（568M encoder，tag 拉取）——用 inference.py
# MODEL_ID = "MaYiding/EventTwin@v1.0"（轻量独立判定）/ @v1.2（零误并闸门）
```

完整示例（单对/批量/结构化模板/级联路由）：[inference_v2.py](inference_v2.py)（v2.0）与
[inference.py](inference.py)（v1.x）。

## Cascade recipe (production)

```text
all pairs → gate: v1.2 (@v1.2, 568M, 2-5ms)  →  p≥0.9 auto-SAME / p≤0.1 auto-DIFF
                       ↓ gray (~30%)
              judge: v2.0 (main, 4B, ~20ms)  →  p≥0.9 SAME / p≤0.1 DIFF
                       ↓ gray (~10%)
              escalate: stronger judge / human review
```

v2.0 alone already escalates only 10.7% — use the two-stage form when false merges are
intolerable (v2.0 has a ~3% confidently-wrong core on clear negatives), or single-stage v2.0
when a 3% false-merge band is acceptable for your review budget.

## Repository structure

```
EventTwin/
├── README.md                  # this file (EN + collapsible ZH)
├── inference.py               # v1.x (568M encoder): single / batch / routing
├── inference_v2.py            # v2.0 (4B decoder): 200-char protocol / batch / routing
├── MODEL_CARD.md              # mirror of the Hugging Face model card
├── VERSIONS.md                # internal ↔ public version mapping (v8 → v1.0)
├── docs/
│   ├── benchmark-protocol.md  # 1,000-pair stratified gold: construction & metrics
│   └── training-notes.md      # 5-generation archive: 3 failures, diagnosis, 3 data laws
└── LICENSE                    # Apache 2.0
```

Model weights (v2.0: 8 GB / v1.x: 1.1 GB) are **not** in this repo — get them from
[🤖 MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin) together with
`calibration.json` (T = 0.824). The dataset lives at
[📦 MaYiding/EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
(1k benchmark + 24k soft-labeled training pairs).

## Versions

MAJOR = new base model / generational leap, MINOR = same-base data/recipe upgrades —
**v2.0** (internal v15_4b_merged): Qwen3-Reranker-4B decoder, LS+R-Drop+EMA, input protocol
= 200-char truncation. AUROC 0.963 / gray 0.974 / ECE 0.105 / agreement 0.918 / 10.7%
escalation; ~3% saturated false-merge core (pair with v1.2 gate if intolerable).
v1.4 (multi-register synthesis), v1.3, v1.2 (zero-FM gate), v1.0 (lightweight standalone),
v1.1 (superseded) all preserved as tags. Full mapping & v2.1 roadmap (saturated-FM core
removal, 8B exploration): [VERSIONS.md](VERSIONS.md).

## Citation

```bibtex
@misc{eventtwin-2026,
  title  = {EventTwin: A Calibrated Chinese Cross-Encoder Family for Same-Event Judgment},
  author = {Ma, Yiding},
  year   = {2026},
  url    = {https://github.com/MaYiding/EventTwin},
  note   = {Soft-label distillation + multi-register synthesis; weights & data on Hugging Face}
}
```

## License

- Code & docs (this repo): [Apache 2.0](LICENSE)
- Model weights: Apache 2.0 (v2.0 base [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B); v1.x base [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — both Apache 2.0)
- Dataset: CC BY-NC 4.0 (contains news excerpts; see the
  [dataset card](https://huggingface.co/datasets/MaYiding/EventTwin-Data))
