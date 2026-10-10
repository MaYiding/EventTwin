---
license: apache-2.0
language:
  - zh
base_model: Qwen/Qwen3-Reranker-4B
library_name: transformers
pipeline_tag: text-classification
tags:
  - cross-encoder
  - reranker
  - event-coreference
  - event-deduplication
  - chinese
  - news-clustering
  - knowledge-distillation
  - calibrated-probabilities
widget:
  - text: 小米YU7正式上市，售价25.35万元起，共推出三款车型
    text_pair: 小米发布YU7系列车型 售价25.35万起
    example_title: "Same event (P ≈ 0.93)"
  - text: 小米YU7正式上市，售价25.35万元起
    text_pair: 特斯拉Model 3 全系降价1.5万元
    example_title: "Different events (P ≈ 0.02)"
---

# EventTwin v2.4 · 中文同事件判定器

**TL;DR (EN)** — EventTwin judges whether two event descriptions refer to the **same real-world
event** (cross-document event coreference / news deduplication) with **temperature-calibrated
probabilities**. **v2.4** is the *data-hygiene + decisiveness* generation: an audit revealed 448
training pairs where a same-entity-different-event channel ("company X raises funding" vs
"company X files for IPO") had been mislabeled SAME; a four-model median gate relabeled them,
and the v2.3 recipe was retrained on the cleaned corpus. Result: **ECE 0.057 (family's best
calibration)**, escalation at default thresholds down from 23.7% to **4.0%** on the hard
benchmark (**[0.2, 0.9] recipe: 2.1% hard / 1.6% production-like**), clear-negative false-merge
0.0%, coverage 95.0%. Cost: overall AUROC −0.3pt (0.983) and false-merge 2.8%→3.3% — for the
absolute-best discriminator keep [v2.3](./tree/v2.3); for calibration + cost, take v2.4.
Data: [EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data).

---

## 这个模型做什么

输入**两个中文事件描述**，输出**"同一现实事件"的概率**（0-1，已校准）。用于新闻流式
聚类的核心判定："这两条报道说的是同一件事吗"——同一动作的不同媒体报道=同事件；
同产品两次调价/宣布与交割=不同事件。

## ⚠️ 输入协议（v2.x 必读）

v2.0 的官方推理协议：**每侧文本截断到 200 字符**，再套 Qwen3-Reranker 官方模板，
取序列末位 yes/no 双 logit，除温度 T=0.7264 后 softmax（v2.4）。

```python
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("MaYiding/EventTwin")
tok.padding_side = "left"                                   # ★ 左 padding
model = AutoModelForCausalLM.from_pretrained("MaYiding/EventTwin").eval()
T = 0.854                                                   # calibration.json

PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
       "Query. Only give me the judgment and do not output any other words or explanations. "
       'The judgment should be "yes" or "no".<|im_end|>\n<|im_start|>user\nQuery: ')
SFX, SFX2 = "\nDocument: ", "\nJudgment: <|im_end|>\n<|im_start|>assistant\n"
yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
no_id  = tok("no",  add_special_tokens=False)["input_ids"][0]

def same_event_prob(a: str, b: str) -> float:
    a, b = a[:200], b[:200]                                 # ★ 200 字截断（官方协议）
    inp = tok(f"{PFX}{a}{SFX}{b}{SFX2}", return_tensors="pt", truncation=True,
              max_length=512, add_special_tokens=False)
    with torch.no_grad():
        la = model(**inp).logits[:, -1, :].float()          # ★ 左 padding 下 -1 即末位
    two = torch.stack([la[:, no_id], la[:, yes_id]], dim=-1)
    return float(torch.softmax(two / T, dim=-1)[0, 1])

same_event_prob("小米YU7正式上市，售价25.35万元起，共推出三款车型",
                "小米发布YU7系列 售价25.35万起")                # ≈ 0.93
```

**为什么必须截断**：实测同一权重，全文输入（均长 135 字、9% 超 200 字）总体 AUROC
0.834，200 字截断 **0.963**（+13pt）——长尾证据/范围后缀会让 decoder 分心（与判定模型
普遍的 context-rot 现象同族）。训练数据文本本身均长 ~135 字，截断即回到训练分布。
批量推理与级联路由示例：[GitHub inference_v2.py](https://github.com/MaYiding/EventTwin)。

## 版本对比：v1.x 家族 vs v2.0 - v2.4

基准 = 1,000 对双教师确认金标（[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)）；
v1.x 系列为 568M encoder 基座（bge-reranker-v2-m3），v2.x 为 **4B decoder 基座**
（Qwen3-Reranker-4B）。所有版本可从本仓库获取（v2.4 = main；v2.3/v2.2/v2.1/v2.0/v1.2/v1.1/v1.0 为 tag）。

### 判别力

| 指标 | v1.0 | v1.2 | v2.1（tag） | v2.2（tag） | v2.3（tag） | **v2.4（main）** |
|---|---|---|---|---|---|---|
| **总体 AUROC** | 0.909 | 0.903 | 0.983 | **0.985** | **0.985** | 0.983 |
| **灰带 AUROC**（最难层） | 0.866 | 0.793 | 0.982 | 0.976 | **0.982** | 0.979 |
| **pos AUROC**（措辞多样同事件） | 0.767 | 0.847 | 0.926 | **0.949** | 0.938 | 0.924 |
| 总体 ECE | 0.187 | 0.261 | 0.076 | 0.150 | 0.094 | **0.057** |
| 与教师一致率 | 0.796 | 0.718 | 0.951 | 0.944 | **0.969** | 0.962 |
| 基座 / 参数 | 568M | 568M | 4B | 4B | 4B | **4B** |

（v2.0 0.963/v1.1 0.895/v1.3 0.902 已略；完整数据见 GitHub 训练档案。）

### 置信分带行为（级联生产口径）

**真值口径**（按 1,000 对终版金标真值统计）：

| 指标 | **v2.1（tag）** | **v2.2（tag）** | **v2.3（tag）** | **v2.4（main）** |
|---|---|---|---|---|
| 误自动并率（真负例≥0.9 / ≥0.95） | 2.9% / 2.8% | 3.7% / 3.2% | **2.8% / 2.8%** | 3.3% / 3.3% |
| 明确负例误并（neg 层≥0.9） | 0.0% | 0.0% | 0.0% | **0.0%** |
| 自动并覆盖（真正例≥0.9） | 91.3% | **96.7%** | 95.4% | 95.0% |
| 漏并率（真正例≤0.1） | 1.7% | 0.0% | **0.0%** | 0.8% |
| 灰带升级率（0.1-0.9，难例分层集） | 13.4% | 56.6% | 23.7% | **4.0%** |

**双流量画像**（升级率取决于流量难度，选型前先看这条）：

| 流量画像 | 模型 | 误并 | 漏并 | 升级率@默认 | 升级率@[0.2,0.9] |
|---|---|---|---|---|---|
| 最坏情形（benchmark 分层难例） | v2.3 | **2.8%** | **0.0%** | 23.7% | 4.3% |
| | **v2.4** | 3.3% | 0.8% | **4.0%** | **2.1%** |
| 自然难度（生产 val 口径） | v2.3 | 0.4% | 0.4% | 9.5% | 2.7% |
| | **v2.4** | **0.4%** | 0.4% | 4.1% | **1.6%** |

v2.4 的核心价值是**果断度与校准**：默认阈值下升级率 4.0%（v2.3 的 1/6），ECE 0.057
历代最佳——分数可直接当置信度用。代价：金标 AUROC −0.3pt、误并 2.8→3.3%、漏并
0→0.8%。追求金标单点最优选 v2.3（tag），成本/校准敏感选 **v2.4（main）**。

**生产阈值建议（重要）**：把自动 DIFF 阈值从 0.10 抬到 **0.20**（即 [0.2, 0.9]），
v2.4 升级率降为**难例 2.1% / 自然流量 1.6%**，误并不变（3.3%）、漏并 1.2%——
0.1-0.2 分数带几乎全是真负例。t_high 不要低于 0.9（硬负例集中在 0.3-0.9 带，
下探即误并翻倍）。各档位与被否决的集成方案实测数字见
[GitHub docs/escalation-recipes.md](https://github.com/MaYiding/EventTwin)。

### 选型建议

- **v2.4（main）· 校准+果断旗舰**：ECE 0.057 历代最佳（分数即置信度）+ 默认升级率
  4.0%（v2.3 的 1/6）+ [0.2,0.9] 配方下 2.1%/1.6%——成本敏感、概率直接用于路由的
  默认之选。**注意**：金标判别力 −0.3pt、误并 2.8→3.3%、漏并 0→0.8%；
- **v2.3（tag）· 金标鲁棒王**：AUROC 0.985/误并 2.8%/漏并 0%——对抗流量、误并敏感
  场景的最优单模；
- **v1.2（tag）· 零误并闸门**：高置信带 0% 误并——前置保险或独立安全闸；
- **v1.0（tag）· 轻量独立判定**：568M、无 GPU 预算或边缘部署；
- v2.2（tag）：跨粒度/覆盖极致档；v2.1（tag）：历史校准档；其余为档案。

对照参考：教师（商业判定 API，闭源）0.998 / ECE 0.062；通用 embedding-8B 0.977 但
**ECE 0.563**（分数不可当概率用）。

## v2.4 训练方法

- **基座**（历代相同）：[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
  （decoder 式 yes/no logit 打分）+ LoRA(r=32) + **权重合并发布**
- **数据卫生（本版核心）**：全量标签审计发现 `lineage` 通道（同主体事件谱系对）448 对
  被错标 SAME（如"完成融资 vs 计划 IPO"、"财报 vs 单品销量"——同主体**不同**事件）。
  用 v23b/v24b/v25/v26 四模型中位数门控重标：中位数 ≥0.7 保留 39 对，其余 409 对改 0.02。
  v2.3 配方在清洗后的语料上重训
- **其余与 v2.3 相同**：LLM 复核精化软标签 37k + 四象限各 1.5k + 双教师共识合成 12k；
  LS=0.05 + 锐化 [0.02, 0.98] + R-Drop KL=2.0 + EMA + lr 1e-4 · 2 epoch
- **温度后校准**：T=0.7264
- **算力**：1× RTX 4090D 24G，约 5.5 小时

## v2.4 基准结果（200 字截断协议）

| 层 | n | AUROC | ECE |
|---|---|---|---|
| OVERALL | 1000 | 0.9826 | **0.057** |
| 灰带（最难） | 300 | 0.9791 | — |
| pos（措辞多样同事件） | 300 | 0.9237 | — |
| 明确负 | 400 | —（带行为见分带表：≥0.9 误并 0.0%） | — |

## 版本说明

公开版本号与内部迭代代号解耦：v1.0=v8 / v1.1=v9 / v1.2=v10a / v2.0=v15_4b_merged /
v2.1=v23b_4b_merged / v2.2=v24b_4b_merged / v2.3=v26_4b_merged / **v2.4=v29_4b_merged**
（v1.3/v1.4 权重因当时的网络拦截未能上传，配方与数据完整入档 EventTwin-Data，
指标见 GitHub VERSIONS）。**MAJOR 位 = 换基模或代际跃升**；MINOR 位 = 同基模数据/配方
升级。v2.4 的教训入档：评测集标签噪声可把"漏并率"污染 3-8 倍并误导整条阈值研究线——
分带指标必须按数据通道分解后再下结论（详见 GitHub VERSIONS）。

## 局限（如实）

- **误并 3.3%**（真值口径，v2.3 为 2.8%）：明确负例零误并，但真值存疑的模糊对仍会
  打到 ≥0.9——绝对零误并场景用 v2.3 + 双级复核；
- 漏并 0.8%（v2.3 为 0.0%）：[0.2,0.9] 配方下 1.2%；
- 金标 AUROC −0.3pt / pos 层 0.924（v2.2 0.949）；
- 输入必须遵循 200 字截断协议（全文输入 AUROC 大跌，v2.0 实测）；
- 域偏移：训练语料为企业新闻域（合成扩产进行中）；4B 权重 8GB（CPU 不可用）。

## 引用

```bibtex
@misc{eventtwin-2026,
  title  = {EventTwin: A Calibrated Chinese Cross-Encoder Family for Same-Event Judgment},
  author = {Ma, Yiding},
  year   = {2026},
  url    = {https://huggingface.co/MaYiding/EventTwin},
  note   = {v2.0: Qwen3-Reranker-4B + distillation + multi-register synthesis; data: MaYiding/EventTwin-Data}
}
```

## 许可

Apache 2.0（基座 Qwen3-Reranker-4B 与 bge-reranker-v2-m3 均为 Apache 2.0）。
训练与评测数据在 [EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)（CC BY-NC 4.0）。
