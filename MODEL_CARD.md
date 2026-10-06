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

# EventTwin v2.1 · 中文同事件判定器

**TL;DR (EN)** — EventTwin judges whether two event descriptions refer to the **same real-world
event** (cross-document event coreference / news deduplication) with **temperature-calibrated
probabilities**. **v2.1** keeps the [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
base and upgrades the data recipe: every teacher soft label is **re-adjudicated by a strong LLM**
(agree→sharpen, disagree→flip), plus 3.5k cross-granularity pairs (event frame × full news text).
Result: **overall AUROC 0.983** — a *single model* now beats the previous three-model ensemble
(0.981) — with **gray-zone 0.982**, **pos (multi-wording) 0.926 (+7.7pt)**, **ECE 0.076**
(approaching the teacher's 0.062), and the v2.0 saturated false-merge core **eliminated**
(clear-negative FM 3.0%→**0.0%**; truth-based FM 8.0%→**2.9%**). Trade-off: gray-band
escalation rises to 13.4%. All prior profiles remain as tags ([v2.0](./tree/v2.0) /
[v1.2](./tree/v1.2) / v1.1 / v1.0). Data:
[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data).

---

## 这个模型做什么

输入**两个中文事件描述**，输出**"同一现实事件"的概率**（0-1，已校准）。用于新闻流式
聚类的核心判定："这两条报道说的是同一件事吗"——同一动作的不同媒体报道=同事件；
同产品两次调价/宣布与交割=不同事件。

## ⚠️ 输入协议（v2.x 必读）

v2.0 的官方推理协议：**每侧文本截断到 200 字符**，再套 Qwen3-Reranker 官方模板，
取序列末位 yes/no 双 logit，除温度 T=0.733 后 softmax（v2.1）。

```python
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("MaYiding/EventTwin")
tok.padding_side = "left"                                   # ★ 左 padding
model = AutoModelForCausalLM.from_pretrained("MaYiding/EventTwin").eval()
T = 0.733                                                   # calibration.json

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

## 版本对比：v1.x 家族 vs v2.0 / v2.1

基准 = 1,000 对双教师确认金标（[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)）；
v1.x 系列为 568M encoder 基座（bge-reranker-v2-m3），v2.x 为 **4B decoder 基座**
（Qwen3-Reranker-4B）。所有版本可从本仓库获取（v2.1 = main；v2.0/v1.2/v1.1/v1.0 为 tag）。

### 判别力

| 指标 | v1.0 | v1.2 | **v2.0（tag）** | **v2.1（main）** | v2.1 提升 |
|---|---|---|---|---|---|
| **总体 AUROC** | 0.909 | 0.903 | 0.963 | **0.983** | **+2.0pt**（超旧三模型集成 0.981） |
| **灰带 AUROC**（最难层） | 0.866 | 0.793 | 0.974 | **0.982** | +0.8pt |
| **pos AUROC**（措辞多样同事件） | 0.767 | 0.847 | 0.850 | **0.926** | **+7.7pt** |
| 总体 ECE | 0.187 | 0.261 | 0.105 | **0.076** | 历代最优（教师 0.062） |
| 与教师一致率 | 0.796 | 0.718 | 0.918 | **0.951** | 逼近教师噪声上界 |
| 基座 / 参数 | 568M encoder | 568M | 4B decoder | **4B decoder** | 同基座配方升级 |

（v1.1 0.895/v1.3 0.902 已略；完整六版数据见 GitHub 训练档案。）

### 置信分带行为（级联生产口径）

**真值口径**（按 1,000 对终版金标真值统计——用户实际体验的误并/漏并；v2.1 起新主口径）：

| 指标 | **v2.0** | **v2.1（main）** |
|---|---|---|
| 误自动并率（真负例≥0.9 / ≥0.95） | 8.0% / 7.9% | **2.9% / 2.8%** |
| 明确负例误并（neg 层≥0.9） | 3.0% | **0.0%**（饱和误并核消除） |
| 自动并覆盖（真正例≥0.9） | 92.5% | 91.3% |
| 漏并率（真正例≤0.1） | 0.0% | 1.7% |
| 灰带升级率（0.1-0.9） | 10.7% | 13.4% |

**层口径**（按构建层统计，与 v1.x 历史表可比；pos 构建层 35% 对的终标为否，
故该口径的"漏并/覆盖"含构造标签噪声，仅作纵向对照）：

| 指标 | v1.0 | v1.2 | **v2.0** | **v2.1** |
|---|---|---|---|---|
| 误自动并率（neg层≥0.9） | 0.25% | **0.0%** | 3.0% | **0.0%** |
| 漏并率（pos层≤0.1） | 6.3% | 3.7% | 13.0% | 14.7% |
| 自动并覆盖（pos层≥0.9） | 76.3% | 28.0% | 73.3% | 64.7% |
| 灰带升级率（0.1-0.9） | 15.6% | 70.3% | 10.7% | 13.4% |

### 选型建议

- **v2.1（main）· 新旗舰**：判别力/校准/安全性全面最优（AUROC 0.983 + 明确负例零误并 +
  ECE 0.076）——排序、终审、自动并的默认之选，且不再需要三模型集成（单模型即超旧集成
  0.981）。**注意**：真值口径仍有 2.9% 灰带区误并（源自真值存疑的模糊对，非明确负例）、
  漏并 1.7%；误并绝对零容忍场景仍建议 v1.2 前置；
- **v1.2 · 零误并闸门**：高置信带 0% 误并 + 3.7% 漏并——作 v2.0 前置保险或独立安全闸；
- **v1.0 · 轻量独立判定**：568M、ECE 0.187，无 GPU 预算或边缘部署；
- v1.1/v1.3/v1.4：历史档案（v1.4 的多语域合成数据遗产已汇入 v2.0 训练集）。

对照参考：教师（商业判定 API，闭源）0.998 / ECE 0.062；通用 embedding-8B 0.977 但
**ECE 0.563**（分数不可当概率用）。

## v2.1 训练方法

- **基座**（与 v2.0 相同）：[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
  （decoder 式 yes/no logit 打分）+ LoRA(r=32) + **权重合并发布**
- **数据升级（本版核心）**：v2.0 的 51,685 对软标签逐对经**强 LLM 独立复核精化**——
  判定方向与教师一致 → 置信提升（clip [0.05, 0.95]）；分歧 → 翻转为 LLM 判定；
  另增 3.5k **跨粒度对**（结构化事件框架 × 新闻正文，真实语料）。复核判定的抽取做了
  三重加固（判定行正则最后匹配优先、方向词负向后顾、无判定重试不默认判否）——
  修复该抽取缺陷本身带来 +1.9pt。见 [EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
  train-pairs-55k
- **正则**：标签平滑 LS=0.1 + R-Drop KL=2.0 + EMA 权重平均 + lr 1e-4 · 2 epoch ·
  batch 8 · 梯度检查点（4B 在 24G 卡上训练的必要开关）
- **温度后校准**：T=0.733
- **算力**：1× RTX 4090D 24G，约 4 小时
- 4B 路线前史：内部 v3（4B+旧配方，0.914 持平 568M）与 v15 未合并 adapter 的误评测
  曾两次"证伪"该路线——v2.0 证明**换基模的收益要配新配方（更强正则+更高 lr）才能兑现**，
  完整复盘见 GitHub 训练档案。

## v2.1 基准结果（200 字截断协议）

| 层 | n | AUROC | ECE |
|---|---|---|---|
| OVERALL | 1000 | **0.9829** | **0.076** |
| 灰带（最难） | 300 | **0.9817** | — |
| pos（措辞多样同事件） | 300 | **0.9264** | — |
| 明确负 | 400 | —（带行为见分带表：≥0.9 误并 0.0%） | — |

## 版本说明

公开版本号与内部迭代代号解耦：v1.0=v8 / v1.1=v9 / v1.2=v10a / **v2.0=v15_4b_merged** /
**v2.1=v23b_4b_merged**（v1.3/v1.4 权重因当时的网络拦截未能上传，配方与数据完整入档
EventTwin-Data，指标见 GitHub VERSIONS）。**MAJOR 位 = 换基模或代际跃升**；MINOR 位 =
同基模数据/配方升级。v2.0 遗留的 3% 饱和误并核在 v2.1 通过标签精化消除（明确负例
0%）；下一步（v2.2）：四象限扩产（短句×簇卡 / 长文×长文 / 长文×簇卡，事件居中节选 +
硬负例挖掘），目标灰带 ≥0.985。

## 局限（如实）

- **灰带区误并 2.9%**（真值口径）：明确负例已零误并，但真值存疑的模糊对仍有 ~2.9%
  会打到 ≥0.9——绝对零误并场景仍建议 v1.2 前置或双级复核；
- 漏并率 1.7%（真正例≤0.1），略高于 v2.0 的 0%；
- 升级率 13.4%（v2.0 为 10.7%）——果断度小幅回退换安全性；
- 输入必须遵循 200 字截断协议（全文输入 AUROC 掉 13pt，v2.0 实测）；
- 措辞多样同事件对（pos 0.926）仍低于教师（0.99）；
- 域偏移：仅企业新闻域；4B 权重 8GB，部署成本高于 568M 系（CPU 不可用）。

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
