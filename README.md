# EventTwin

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-EventTwin%20v2.4-yellow)](https://huggingface.co/MaYiding/EventTwin)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EventTwin--Data-orange)](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20Transformers-red)](https://pytorch.org)
[![Language](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](#-中文文档)

**EventTwin** (latest: **v2.4**) judges whether two Chinese event descriptions or documents refer to the
**same real-world event**, and answers with a **temperature-calibrated probability** (not a
similarity score) — usable directly as confidence for routing, gating, and cascade pipelines.

**v2.4** is the *data-hygiene + decisiveness* generation: a full label audit uncovered 448
training pairs from a same-entity lineage channel that were mislabeled SAME ("raises funding"
vs "files for IPO" — same company, different events); a four-model median gate relabeled them
and the v2.3 recipe was retrained on the cleaned corpus. Result: **ECE 0.057 (family's best
calibration)**, default-band escalation down from 23.7% to **4.0%** on the hard stratified
benchmark (**[0.2, 0.9] recipe: 2.1% hard / 1.6% production-like**), clear-negative FM 0.0%.
Trade-off: overall AUROC −0.3pt and truth-FM 2.8%→3.3% — the absolute-best discriminator
remains [v2.3](../../tree/v2.3); **v1.2** remains the zero-FM gate; **v1.0** the lightweight
standalone (568M). Full decision matrix and rejected alternatives (consensus gating, median
ensembles, cascade rescoring): [docs/escalation-recipes.md](docs/escalation-recipes.md).

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
all pairs → judge: v2.4 (main, 4B, ~20ms) → p≥0.90 auto-SAME / p≤0.20 auto-DIFF
                       ↓ gray (~2-4%)
              escalate: stronger judge / human review
```

**Threshold tuning is the cheapest lever you have.** v2.4 already escalates only 4.0% of hard
traffic at the default [0.1, 0.9] band (v2.3: 23.7%); widening the auto-DIFF band to
**[0.2, 0.9]** cuts it to **2.1%** (hard gold) / **1.6%** (production-like traffic) with FM
unchanged at 3.3% and miss at 1.2% — the 0.1-0.2 score band is almost entirely true negatives.
Do **not** lower t_high below 0.9: hard negatives (near-duplicate wording, different events)
concentrate in 0.3-0.9 and start merging. Full profile-by-profile table and rejected
alternatives (consensus gating, median ensembles, cascade rescoring):
[docs/escalation-recipes.md](docs/escalation-recipes.md).

If even gray-zone ambiguity must not auto-merge, prepend the v1.2 gate (568M, 2-5 ms), or use
[v2.3](../../tree/v2.3) whose truth-FM (2.8%) is the family's lowest single model.

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
│   ├── escalation-recipes.md  # production threshold guide: [0.2,0.9] cuts escalation 75-82%
│   └── training-notes.md      # 5-generation archive: 3 failures, diagnosis, 3 data laws
└── LICENSE                    # Apache 2.0
```

Model weights (v2.4: 8 GB / v1.x: 1.1 GB) are **not** in this repo — get them from
[🤖 MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin) together with
`calibration.json` (v2.4 T = 0.7264). The dataset lives at
[📦 MaYiding/EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
(1k benchmark + 55k v2.1 pairs + 60k v2.2 quadrants recipe + 75k v2.3 synth recipe).

## Versions

MAJOR = new base model / generational leap, MINOR = same-base data/recipe upgrades —
**v2.4** (internal v29_4b_merged): v2.3 recipe + lineage label hygiene (448 mislabeled
same-entity-different-event pairs, four-model median gate, 409 relabeled); **ECE 0.057
(family's best)**, default escalation 23.7%→**4.0%** ([0.2,0.9]: 2.1%/1.6%), clear-neg FM
0.0%; cost: AUROC −0.3pt, truth-FM 2.8→3.3%, miss 0→0.8%.
**v2.3** (tag): the discriminator king (AUROC 0.985, truth-FM 2.8%, miss 0%).
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
  note   = {v2.4: label hygiene + decisiveness; weights & data on Hugging Face}
}
```

## License

- Code & docs (this repo): [Apache 2.0](LICENSE)
- Model weights: Apache 2.0 (v2.x base [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B); v1.x base [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — both Apache 2.0)
- Dataset: CC BY-NC 4.0 (contains news excerpts; see the
  [dataset card](https://huggingface.co/datasets/MaYiding/EventTwin-Data))
