# EventTwin

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-EventTwin%20v2.1-yellow)](https://huggingface.co/MaYiding/EventTwin)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EventTwin--Data-orange)](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20Transformers-red)](https://pytorch.org)
[![Language](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](#-中文文档)

**EventTwin** (latest: **v2.1**) judges whether two Chinese event descriptions refer to the
**same real-world event**, and answers with a **temperature-calibrated probability** (not a
similarity score) — usable directly as confidence for routing, gating, and cascade pipelines.

**v2.1** keeps the Qwen3-Reranker-4B base and upgrades the data recipe: every teacher soft
label **re-adjudicated by a strong LLM** (agree→sharpen / disagree→flip) plus cross-granularity
pairs (event frame × full news text). A *single model* now beats the previous three-model
ensemble: **overall AUROC 0.983** (ensemble 0.981, v2.0 0.963), **gray-zone 0.982**,
**pos (multi-wording) 0.926 (+7.7pt)**, **ECE 0.076**, and the v2.0 saturated false-merge core
**eliminated** (clear-negative FM 3.0%→0.0%; truth-based FM 8.0%→2.9%). Trade-off: escalation
10.7%→13.4%. **v1.2** remains the zero-false-merge gate; **v1.0** the lightweight standalone
(568M, 2-5 ms/pair; 4B ~20 ms/pair). All on a 1,000-pair dual-teacher gold with full benchmark
protocol, training data, and a training archive (including failed iterations and data laws).

## Quick start

```bash
pip install torch transformers
```

```python
# v2.1（main，4B decoder）——三条协议红线：200字截断 / 左padding取末位 / 除温度T
from inference_v2 import load, same_event_prob   # 本仓库 inference_v2.py
tok, model, T, yes_id, no_id = load("MaYiding/EventTwin")
same_event_prob("小米YU7正式上市，售价25.35万元起，共推出三款车型",
                "小米发布YU7系列 售价25.35万起")          # ≈ 0.93

# v1.x（568M encoder，tag 拉取）——用 inference.py
# MODEL_ID = "MaYiding/EventTwin@v1.0"（轻量独立判定）/ @v1.2（零误并闸门）
```

完整示例（单对/批量/结构化模板/级联路由）：[inference_v2.py](inference_v2.py)（v2.x）与
[inference.py](inference.py)（v1.x）。

## Cascade recipe (production)

```text
all pairs → gate: v1.2 (@v1.2, 568M, 2-5ms)  →  p≥0.9 auto-SAME / p≤0.1 auto-DIFF
                       ↓ gray (~30%)
              judge: v2.1 (main, 4B, ~20ms)  →  p≥0.9 SAME / p≤0.1 DIFF
                       ↓ gray (~13%)
              escalate: stronger judge / human review
```

v2.1 alone escalates 13.4% with **zero false merges on clear negatives** (0.0% at ≥0.9) —
use the two-stage form only when even gray-zone ambiguity must not auto-merge (truth-based
FM 2.9% comes from genuinely ambiguous pairs), or single-stage v2.1 otherwise.

## Repository structure

```
EventTwin/
├── README.md                  # this file (EN + collapsible ZH)
├── inference.py               # v1.x (568M encoder): single / batch / routing
├── inference_v2.py            # v2.x (4B decoder): 200-char protocol / batch / routing
├── MODEL_CARD.md              # mirror of the Hugging Face model card
├── VERSIONS.md                # internal ↔ public version mapping (v8 → v1.0)
├── docs/
│   ├── benchmark-protocol.md  # 1,000-pair stratified gold: construction & metrics
│   └── training-notes.md      # 5-generation archive: 3 failures, diagnosis, 3 data laws
└── LICENSE                    # Apache 2.0
```

Model weights (v2.1: 8 GB / v1.x: 1.1 GB) are **not** in this repo — get them from
[🤖 MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin) together with
`calibration.json` (v2.1 T = 0.733). The dataset lives at
[📦 MaYiding/EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
(1k benchmark + 55k soft-labeled training pairs).

## Versions

MAJOR = new base model / generational leap, MINOR = same-base data/recipe upgrades —
**v2.1** (internal v23b_4b_merged): LLM-refined soft labels + cross-granularity pairs;
AUROC **0.983** (beats the old 3-model ensemble 0.981) / gray 0.982 / pos 0.926 / ECE 0.076 /
clear-negative FM 0.0% / truth-based FM 2.9% / escalation 13.4%.
**v2.0** (tag): the 4B-decoder generational leap (0.963). **v1.2** (tag): zero-FM gate;
**v1.0** (tag): lightweight standalone; v1.1 (superseded). v1.3/v1.4 weights never uploaded
(network-blocked at the time; recipes & data archived in the dataset, metrics in VERSIONS).
Full mapping & v2.2 roadmap (four-quadrant cross-granularity scaling): [VERSIONS.md](VERSIONS.md).

## Citation

```bibtex
@misc{eventtwin-2026,
  title  = {EventTwin: A Calibrated Chinese Cross-Encoder Family for Same-Event Judgment},
  author = {Ma, Yiding},
  year   = {2026},
  url    = {https://github.com/MaYiding/EventTwin},
  note   = {v2.1: LLM-refined soft-label distillation; weights & data on Hugging Face}
}
```

## License

- Code & docs (this repo): [Apache 2.0](LICENSE)
- Model weights: Apache 2.0 (v2.x base [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B); v1.x base [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — both Apache 2.0)
- Dataset: CC BY-NC 4.0 (contains news excerpts; see the
  [dataset card](https://huggingface.co/datasets/MaYiding/EventTwin-Data))
