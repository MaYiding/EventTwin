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

# EventTwin v2.0 · 中文同事件判定器

**TL;DR (EN)** — EventTwin judges whether two event descriptions refer to the **same real-world
event** (cross-document event coreference / news deduplication) with **temperature-calibrated
probabilities**. **v2.0** is the first release on a new base —
[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) (decoder, yes/no-logit) —
and the family's biggest leap: **overall AUROC 0.963** (v1.x best: 0.909), **gray-zone AUROC
0.974** (the long-standing gap vs v1.0's 0.866 closed), **ECE 0.105**, agreement-with-teacher
**0.918**, and the most decisive band behavior yet (only **10.7%** gray-band escalation,
72.7% auto-merge coverage). Trade-off: ~3% of clear negatives score ≥0.9 (confidently-wrong
core) — if you need zero false merges, keep [v1.2](./tree/v1.2) as the gate. All five v1.x
profiles remain available as tags. Data:
[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data).

---

## 这个模型做什么

输入**两个中文事件描述**，输出**"同一现实事件"的概率**（0-1，已校准）。用于新闻流式
聚类的核心判定："这两条报道说的是同一件事吗"——同一动作的不同媒体报道=同事件；
同产品两次调价/宣布与交割=不同事件。

## ⚠️ 输入协议（v2.0 必读）

v2.0 的官方推理协议：**每侧文本截断到 200 字符**，再套 Qwen3-Reranker 官方模板，
取序列末位 yes/no 双 logit，除温度 T=0.6856 后 softmax。

```python
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("MaYiding/EventTwin")
tok.padding_side = "left"                                   # ★ 左 padding
model = AutoModelForCausalLM.from_pretrained("MaYiding/EventTwin").eval()
T = 0.6856                                                  # calibration.json

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

## 版本对比：v1.x 家族 vs v2.0

基准 = 1,000 对双教师确认金标（[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)）；
v1.x 系列为 568M encoder 基座（bge-reranker-v2-m3），v2.0 为 **4B decoder 基座**
（Qwen3-Reranker-4B）。所有版本可从本仓库获取（v2.0 = main；v1.4/v1.3/v1.2/v1.1/v1.0 为 tag）。

### 判别力

| 指标 | v1.0 | v1.2 | v1.4 | **v2.0（main）** | 提升 |
|---|---|---|---|---|---|
| **总体 AUROC** | 0.909 | 0.903 | 0.900 | **0.963** | **+5.4pt** |
| **灰带 AUROC**（最难层） | 0.866 | 0.793 | 0.788 | **0.974** | **+10.8pt** |
| **pos AUROC**（措辞多样同事件） | 0.767 | 0.847 | 0.843 | **0.850** | 历代最高 |
| 灰带 acc | 0.853 | 0.883 | 0.883 | **0.947** | +6.4pt |
| 总体 ECE | 0.187 | 0.261 | 0.304 | **0.105** | 历代最优 |
| 与教师一致率 | 0.796 | 0.718 | 0.684 | **0.918** | 逼近教师噪声上界 |
| 基座 / 参数 | 568M encoder | 568M | 568M | **4B decoder** | 换代 |

（v1.1 0.895/v1.3 0.902 已略；完整六版数据见 GitHub 训练档案。）

### 置信分带行为（级联生产口径）

| 指标 | v1.0 | v1.2 | v1.4 | **v2.0** |
|---|---|---|---|---|
| 误自动并率（neg≥0.9 / ≥0.95） | 0.25% / — | **0.0%** / — | 1.25% / — | 3.25% / 3.0% |
| 漏并率（pos≤0.1） | 6.3% | 3.7% | 12.0% | 13.0% |
| 自动并覆盖（pos≥0.9） | 76.3% | 28.0% | 70.7% | **72.7%** |
| 灰带升级率（0.1-0.9） | 15.6% | 70.3% | 30.7% | **10.7%** |

### 选型建议

- **v2.0（main）· 新旗舰**：判别力/校准/果断度全面最优（10.7% 升级率 + 72.7% 覆盖 +
  AUROC 0.963）——排序、终审、中等预算自动并的默认之选。**注意**：~3% 的明确负例会被
  高置信误并（十二三个饱和错例），误并零容忍的自动归并闸门请用 v1.2 前置或提高阈值并
  配第二级复核；
- **v1.2 · 零误并闸门**：高置信带 0% 误并 + 3.7% 漏并——作 v2.0 前置保险或独立安全闸；
- **v1.0 · 轻量独立判定**：568M、ECE 0.187，无 GPU 预算或边缘部署；
- v1.1/v1.3/v1.4：历史档案（v1.4 的多语域合成数据遗产已汇入 v2.0 训练集）。

对照参考：教师（商业判定 API，闭源）0.998 / ECE 0.062；通用 embedding-8B 0.977 但
**ECE 0.563**（分数不可当概率用）。

## v2.0 训练方法

- **基座**：[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
  （decoder 式 yes/no logit 打分，MMTEB-R 72.74）+ LoRA(r=32) + **权重合并发布**
- **数据**：与 v1.4 同源（51,685 对，正 13,425——含 10 人设多语域合成 ~8k、提及级通道），
  见 [EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)
- **正则**：标签平滑 LS=0.1 + R-Drop KL=2.0 + EMA 权重平均 + lr 1e-4 · 2 epoch ·
  batch 8 · 梯度检查点（4B 在 24G 卡上训练的必要开关）
- **温度后校准**：T=0.6856（val n=5,168）
- **算力**：1× RTX 4090D 24G，约 4 小时
- 4B 路线前史：内部 v3（4B+旧配方，0.914 持平 568M）与 v15 未合并 adapter 的误评测
  曾两次"证伪"该路线——v2.0 证明**换基模的收益要配新配方（更强正则+更高 lr）才能兑现**，
  完整复盘见 GitHub 训练档案。

## v2.0 基准结果（200 字截断协议）

| 层 | n | AUROC | acc@最优阈 | ECE |
|---|---|---|---|---|
| OVERALL | 1000 | **0.9633** | 0.930 | **0.105** |
| 灰带（最难） | 300 | **0.9739** | 0.947 | 0.095 |
| pos（措辞多样同事件） | 300 | **0.8495** | 0.853 | 0.148 |
| 明确负 | 400 | — | 0.963 | 0.440（带行为见分带表） |

## 版本说明

公开版本号与内部迭代代号解耦：v1.0=v8 / v1.1=v9 / v1.2=v10a / v1.3=v13 / v1.4=v14 /
**v2.0=v15_4b_merged**。**MAJOR 位 = 换基模或代际跃升**（v2.0：568M encoder → 4B
decoder，总体 AUROC +5.4pt、灰带 +10.8pt）；MINOR 位 = 同基模数据/配方升级。
内部 v16（4B 灰带过采样）因 EMA 显存溢出中止。v2.1 方向：压掉 3% 饱和误并核
（hard-negative 过采样 / 分带蒸馏）、8B 基座探索。

## 局限（如实）

- **~3% 饱和误并核**：12-13 个明确负例被打到 ≥0.95 高分且阈值上探无效——纯高阈值
  自动并场景须配 v1.2 前置或第二级复核；
- 漏并率 13%（pos≤0.1）高于 v1.2 的 3.7%；
- 输入必须遵循 200 字截断协议（全文输入 AUROC 掉 13pt）；
- 措辞多样同事件对（pos 0.85）仍低于教师（0.99）；
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
