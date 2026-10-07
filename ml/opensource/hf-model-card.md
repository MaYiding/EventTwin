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

# EventTwin v2.3 · 中文同事件判定器

**TL;DR (EN)** — EventTwin judges whether two event descriptions refer to the **same real-world
event** (cross-document event coreference / news deduplication) with **temperature-calibrated
probabilities**. **v2.3** keeps the [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) base
and refines the recipe with **sharpened labels + dual-teacher-consensus synthetic data**:
LLM-refined soft labels (v2.1) + four cross-granularity quadrants (v2.2) + 12k multi-register
synthetic pairs (social-media / colloquial / wire / long-form, kept only when the generator's
judgment agrees with a second judge's probability). Result: **overall AUROC 0.985 + gray
0.982 (records)**, **ECE 0.094**, **truth-based false-merge 2.8% (family's lowest)** with
**coverage 95.4% / miss 0.0%**, and escalation down from v2.2's 56.6% to **23.7%** on the
hard stratified benchmark. Quadrant hold-out 0.98-1.0. Prior profiles remain as tags
([v2.2](./tree/v2.2) / [v2.1](./tree/v2.1) / v2.0 / v1.x). Data:
[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data).

---

## 这个模型做什么

输入**两个中文事件描述**，输出**"同一现实事件"的概率**（0-1，已校准）。用于新闻流式
聚类的核心判定："这两条报道说的是同一件事吗"——同一动作的不同媒体报道=同事件；
同产品两次调价/宣布与交割=不同事件。

## ⚠️ 输入协议（v2.x 必读）

v2.0 的官方推理协议：**每侧文本截断到 200 字符**，再套 Qwen3-Reranker 官方模板，
取序列末位 yes/no 双 logit，除温度 T=0.854 后 softmax（v2.3）。

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

## 版本对比：v1.x 家族 vs v2.0 - v2.3

基准 = 1,000 对双教师确认金标（[EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)）；
v1.x 系列为 568M encoder 基座（bge-reranker-v2-m3），v2.x 为 **4B decoder 基座**
（Qwen3-Reranker-4B）。所有版本可从本仓库获取（v2.3 = main；v2.2/v2.1/v2.0/v1.2/v1.1/v1.0 为 tag）。

### 判别力

| 指标 | v1.0 | v1.2 | v2.0（tag） | v2.1（tag） | v2.2（tag） | **v2.3（main）** |
|---|---|---|---|---|---|---|
| **总体 AUROC** | 0.909 | 0.903 | 0.963 | 0.983 | 0.985 | **0.985** |
| **灰带 AUROC**（最难层） | 0.866 | 0.793 | 0.974 | 0.982 | 0.976 | **0.982** |
| **pos AUROC**（措辞多样同事件） | 0.767 | 0.847 | 0.850 | 0.926 | **0.949** | 0.938 |
| **长文×长文 AUROC / acc**（hold-out） | — | — | — | 0.929 / 0.861 | **0.993 / 0.983** | 0.977 / 0.939 |
| **长文×簇卡 acc**（hold-out） | — | — | — | 0.922 | 0.986 | **0.993** |
| 总体 ECE | 0.187 | 0.261 | 0.105 | **0.076** | 0.150 | 0.094 |
| 与教师一致率 | 0.796 | 0.718 | 0.918 | 0.951 | 0.944 | **0.969** |
| 基座 / 参数 | 568M | 568M | 4B | 4B | 4B | **4B** |

（v1.1 0.895/v1.3 0.902 已略；完整六版数据见 GitHub 训练档案。）

### 置信分带行为（级联生产口径）

**真值口径**（按 1,000 对终版金标真值统计）：

| 指标 | **v2.0** | **v2.1（tag）** | **v2.2（tag）** | **v2.3（main）** |
|---|---|---|---|---|
| 误自动并率（真负例≥0.9 / ≥0.95） | 8.0% / 7.9% | 2.9% / 2.8% | 3.7% / 3.2% | **2.8% / 2.8%** |
| 明确负例误并（neg 层≥0.9） | 3.0% | 0.0% | 0.0% | **0.0%** |
| 自动并覆盖（真正例≥0.9） | 92.5% | 91.3% | **96.7%** | 95.4% |
| 漏并率（真正例≤0.1） | 0.0% | 1.7% | 0.0% | **0.0%** |
| 灰带升级率（0.1-0.9，难例分层集） | 10.7% | **13.4%** | 56.6% | 23.7% |

**双流量画像**（升级率取决于流量难度，选型前先看这条）：

| 流量画像 | 模型 | 误并 | 漏并 | 覆盖 | 拒并 | 升级率 |
|---|---|---|---|---|---|---|
| 最坏情形（benchmark 分层难例，30% 灰带构造） | v2.1 | 2.9% | 1.7% | 91.3% | — | **13.4%** |
| | v2.2 | 3.7% | 0.0% | **96.7%** | — | 56.6% |
| | **v2.3** | **2.8%** | **0.0%** | 95.4% | — | **23.7%** |

v2.3 在最坏情形流量上同时拿到低误并（2.8%，历代最低）与中等升级率（23.7%，v2.2 的
四成）；自然难度流量下升级率与 v2.2 同档（val 口径 ~10%）。需要最低升级率（13.4%）
选 v2.1，需要最高覆盖（96.7%）选 v2.2，均衡选 **v2.3**。

**层口径**（按构建层统计，与 v1.x 历史表可比；pos 构建层 35% 对的终标为否，
故该口径的"漏并/覆盖"含构造标签噪声，仅作纵向对照）：

| 指标 | v1.0 | v1.2 | v2.0 | v2.1 | **v2.2** |
|---|---|---|---|---|---|
| 误自动并率（neg层≥0.9） | 0.25% | **0.0%** | 3.0% | 0.0% | **0.0%** |
| 漏并率（pos层≤0.1） | 6.3% | 3.7% | 13.0% | 14.7% | 3.0% |
| 自动并覆盖（pos层≥0.9） | 76.3% | 28.0% | 73.3% | 64.7% | 68.3% |

### 选型建议

- **v2.3（main）· 均衡旗舰**：AUROC/灰带/一致率新高 + 误并 2.8% 历代最低 + 漏并 0% +
  升级率较 v2.2 减半以上 + ECE 0.094——排序、自动并、跨粒度的默认之选。**注意**：
  长文×长文象限较 v2.2 微降（0.977 vs 0.993）、pos 层较 v2.2 低（0.938 vs 0.949）；
  追求单点极致用对应 tag（v2.2 跨粒度/pos，v2.1 校准/果断）；
- **v1.2 · 零误并闸门**：高置信带 0% 误并 + 3.7% 漏并——作 v2.0 前置保险或独立安全闸；
- **v1.0 · 轻量独立判定**：568M、ECE 0.187，无 GPU 预算或边缘部署；
- v1.1/v1.3/v1.4：历史档案（v1.4 的多语域合成数据遗产已汇入 v2.0 训练集）。

对照参考：教师（商业判定 API，闭源）0.998 / ECE 0.062；通用 embedding-8B 0.977 但
**ECE 0.563**（分数不可当概率用）。

## v2.3 训练方法

- **基座**（历代相同）：[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
  （decoder 式 yes/no logit 打分）+ LoRA(r=32) + **权重合并发布**
- **数据升级（本版核心）**：v2.2 配方（LLM 复核精化软标签 37k + 四象限各 1.5k）+
  **双教师共识合成通道 12k**——多模型各生成虚构中小企业/多行业事件的多种表面形式
  （社媒转述/口语/快讯/长文/事件卡），仅保留生成模型判定与第二判定器概率方向一致的
  对（共识过滤），场景级 train/test 切分
- **正则与锐化**：标签平滑 LS=0.05 + 标签极值锐化 [0.02, 0.98] + R-Drop KL=2.0 +
  EMA + lr 1e-4 · 2 epoch · 梯度检查点；**长输入 max_len 1152**
- **温度后校准**：T=0.854
- **算力**：1× RTX 4090D 24G，约 2 小时
- 4B 路线前史：内部 v3（4B+旧配方，0.914 持平 568M）与 v15 未合并 adapter 的误评测
  曾两次"证伪"该路线——v2.0 证明**换基模的收益要配新配方（更强正则+更高 lr）才能兑现**，
  完整复盘见 GitHub 训练档案。

## v2.3 基准结果（200 字截断协议）

| 层 | n | AUROC | ECE |
|---|---|---|---|
| OVERALL | 1000 | **0.9854** | **0.094** |
| 灰带（最难） | 300 | **0.9819** | — |
| pos（措辞多样同事件） | 300 | 0.9379 | — |
| 明确负 | 400 | —（带行为见分带表：≥0.9 误并 0.0%） | — |

**跨粒度 hold-out**（按簇切分防泄漏）：长文×簇卡 0.9954/acc 0.993 / 短句×簇卡
0.9997 / 框架×正文 1.000 / 长文×长文 0.977（v2.2 为 0.993，微降）。

## 版本说明

公开版本号与内部迭代代号解耦：v1.0=v8 / v1.1=v9 / v1.2=v10a / **v2.0=v15_4b_merged** /
**v2.1=v23b_4b_merged / v2.2=v24b_4b_merged / v2.3=v26_4b_merged**（v1.3/v1.4 权重因当时的网络拦截未能上传，
配方与数据完整入档 EventTwin-Data，指标见 GitHub VERSIONS）。**MAJOR 位 = 换基模或
代际跃升**；MINOR 位 = 同基模数据/配方升级。v2.0 遗留的 3% 饱和误并核自 v2.1 起消除
（明确负例 0%）。下一步：合成数据飞轮（中小企业/多行业场景）+ q1 精化全量重训。

## 局限（如实）

- **灰带区误并 2.8%**（真值口径）：明确负例零误并，但真值存疑的模糊对仍会打到 ≥0.9
  ——绝对零误并场景建议 v1.2 前置或双级复核；
- 标签锐化使模型对部分"翻转标签"训练对的拟合下降（val 口径漏并升至 18%，benchmark
  人工金标口径漏并 0.0% 不受影响）——若你的下游标签分布与精化标签强相关，选 v2.2；
- ECE 0.150（v2.1 0.076）——概率直接当置信度用的场景优先 v2.1；
- 输入必须遵循 200 字截断协议（全文输入 AUROC 掉 13pt，v2.0 实测）；
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
