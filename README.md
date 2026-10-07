# EventTwin

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-EventTwin%20v2.3-yellow)](https://huggingface.co/MaYiding/EventTwin)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EventTwin--Data-orange)](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20Transformers-red)](https://pytorch.org)
[![Language](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](#-中文文档)

**EventTwin** (latest: **v2.3**) judges whether two Chinese event descriptions or documents refer to the
**same real-world event**, and answers with a **temperature-calibrated probability** (not a
similarity score) — usable directly as confidence for routing, gating, and cascade pipelines.

**v2.2** adds **cross-granularity judgment** on top of v2.1's LLM-refined distillation: four
training quadrants (event frame × cluster card, article × article, article × card, frame × full
text) teach one model to handle short-pair merging *and* long-text cluster merging/attachment.
**Overall AUROC 0.985** (record) with **pos 0.949** (record), quadrant hold-out **0.99-1.0**
(article×article acc 0.861→**0.983** vs v2.1 zero-shot), **clear-negative FM 0.0%**,
production-traffic escalation 10.7% with FM 0.1% (hard stratified traffic: 56.6% — see the
dual-traffic profile table in the model card). Trade-offs: ECE 0.150 (v2.1: 0.076) and
conservatism on adversarial pairs — see [v2.1](../../tree/v2.1) for the calibration-first
profile; **v1.2** remains the zero-FM gate; **v1.0** the lightweight standalone (568M).

## Quick start

```bash
pip install torch transformers
```

```python
# v2.3（main，4B decoder）——三条协议红线：200字截断 / 左padding取末位 / 除温度T
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

Model weights (v2.3: 8 GB / v1.x: 1.1 GB) are **not** in this repo — get them from
[🤖 MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin) together with
`calibration.json` (v2.3 T = 0.854). The dataset lives at
[📦 MaYiding/EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
(1k benchmark + 55k v2.1 pairs + 60k v2.2 quadrants recipe + 75k v2.3 synth recipe).

## Versions

MAJOR = new base model / generational leap, MINOR = same-base data/recipe upgrades —
**v2.3** (internal v26_4b_merged): v2.2 recipe + sharpened labels (LS 0.05, clip 0.02/0.98)
+ 12k dual-consensus multi-register synthesis; AUROC **0.985** / gray **0.982** (records),
truth-FM **2.8% (lowest)**, escalation 56.6%→**23.7%**, ECE 0.094.
**v2.2** (tag): max coverage 96.7% / pos 0.949. **v2.1** (tag): calibration-first
(ECE 0.076, escalation 13.4%). **v2.0** (tag): the 4B-decoder leap (0.963). **v1.2** (tag):
zero-FM gate; **v1.0** (tag): lightweight standalone. v1.3/v1.4 weights never uploaded
(network-blocked; recipes & data archived in the dataset, metrics in VERSIONS). Full mapping
& roadmap (synthetic SME-domain flywheel): [VERSIONS.md](VERSIONS.md).

## Citation

```bibtex
@misc{eventtwin-2026,
  title  = {EventTwin: A Calibrated Chinese Cross-Encoder Family for Same-Event Judgment},
  author = {Ma, Yiding},
  year   = {2026},
  url    = {https://github.com/MaYiding/EventTwin},
  note   = {v2.3: LLM-refined distillation + quadrants + dual-consensus synthesis; weights & data on Hugging Face}
}
```

## License

- Code & docs (this repo): [Apache 2.0](LICENSE)
- Model weights: Apache 2.0 (v2.x base [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B); v1.x base [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — both Apache 2.0)
- Dataset: CC BY-NC 4.0 (contains news excerpts; see the
  [dataset card](https://huggingface.co/datasets/MaYiding/EventTwin-Data))
