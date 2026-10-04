---
license: cc-by-nc-4.0
language:
  - zh
task_categories:
  - text-classification
tags:
  - event-coreference
  - event-deduplication
  - chinese
  - news-clustering
  - distillation
  - benchmark
size_categories:
  - 1K<n<10K
  - 10K<n<100K
---

# EventTwin-Data · 中文同事件判定数据集（benchmark 1K + 训练对 24K/25K/52K）

**TL;DR (EN)** — Paired Chinese event descriptions with **soft probability labels** for
same-event judgment (cross-document event coreference). Four files:
`benchmark-1k.jsonl` (1,000 stratified pairs, dual-teacher-confirmed gold — the evaluation
set for [MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin));
`train-pairs-52k.jsonl` (51,685 pairs, **v1.2** — used by
[EventTwin v1.4](https://huggingface.co/MaYiding/EventTwin): mention-level expansion +
**10-persona multi-register synthesis** — the same event rewritten as social-media posts,
colloquial retellings, reader comments, research reports, official bulletins, etc.);
`train-pairs-25k.jsonl` (25,458 pairs, **v1.1** — adds a second-generation synthetic
channel with counterfactual filtering) and
`train-pairs-24k.jsonl` (24,308 pairs, **v1.0** — kept for reproducibility). Domain:
enterprise news of 30 Chinese tech/auto/internet companies, 2024-01 ~ 2026-09.

---

## 文件与字段

### 1. `benchmark-1k.jsonl` — 评测金标（1,000 对）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int | 序号（固定抽样种子可复现） |
| `layer` | str | `pos`（明确同事件 300）/ `neg`（明确异事件 400）/ `gray`（灰带 300，最难层） |
| `a`, `b` | str | 两个事件描述（提及文本或簇摘要，含 [类型]/(时间)/证据 结构化后缀） |
| `gold` | int\|null | 金标：1=同一事件，0=不同事件；null=双教师分歧/未确认（评测时的处理见协议） |
| `confirmed` | bool | 是否双教师一致确认 |
| `jev_score` | float | 教师（判定模型）原始概率——供校准/误差分析，**评测时不要用作特征** |

分层构成：

| 层 | n | 构成 | 考验点 |
|---|---|---|---|
| pos | 300 | 多成员簇内提及对（70%）+ 同源转载对（30%） | 措辞多样性与详略变化下的同事件识别 |
| neg | 400 | 跨企业同类型 + 跨类型对（共享实体剔除） | 主题相近干扰的排除 |
| gray | 300 | 同实体×同类型×时间窗重叠（±30 天）的不同簇对 | "同事件"与"同话题相近事件"的边界 |

### 2. `train-pairs-52k.jsonl` — 训练对 v1.2（51,685 对，EventTwin v1.4 训练数据）

相对 v1.1 的增量（正对 13,425，约 2.4 倍）：
- **10 人设多语域合成（~8k）**：同一事件按社交媒体帖/口语转述/读者评论/行业研报/
  自媒体分析/短视频标题/官方通报/新闻通稿/专家点评/财经快讯十种语域改写——
  "措辞多样同事件"覆盖面跃升到跨语域全景（EventTwin v1.3 亦在同族簇级 50k 前身训练）；
- **提及级通道**：簇内提及对（m1_intra 1,046）、仲裁翻正（m2_flip 467）、提及×簇分布对。

### 3. `train-pairs-25k.jsonl` — 训练对 v1.1（25,458 对，EventTwin v1.1/v1.2 训练数据）

字段同下；相对 v1.0 的增量：新增 `c5_synth_v2`（第二代合成改写 726 条，经**反事实过滤**——
LLM 自检改写是否引入新信息，滤除约 300 条不合格）；正对 5,319。EventTwin v1.1（EMA 权重）
即用本文件训练，pos 层 AUROC 0.767→0.837。

### 4. `train-pairs-24k.jsonl` — 训练对 v1.0（24,308 对，EventTwin v1.0 训练数据，保留）

| 字段 | 类型 | 说明 |
|---|---|---|
| `a`, `b` | str | 事件描述对 |
| `label` | float | **软标签**（教师同事件概率；合成对 0.9、翻转对 0.85/0.15）——软标签蒸馏的监督信号 |
| `source` | str | 通道来源（见下表） |
| `split` | str | `train`/`val`/`test`（按时间分层切分，防同日同源泄漏） |

正对 4,593 的通道构成：

| source | n | 说明 |
|---|---|---|
| `c5_synth` | 1,495 | LLM 合成改写（同一事件的"另一家媒体报道"写法，保持主体/对象/时间/动作） |
| `lineage` | 1,482 | 同源转载对（同 lineage_group 不同文档版本） |
| `cluster` | 780 | 多成员簇内提及对 |
| `p_same` | 442 | 教师判定分布的高置信正对 |
| `attach` | 236 | 生产系统归并决策对 |
| `p_same_promoted` | 86 | 疑似负例经教师复核升格的正对 |
| `arb_flip` | 72 | 双教师分歧后仲裁翻转的对 |

负对约 19.7k，以"同实体×同类型×时间窗重叠"hard negative 为主，全部经 **margin 假负例
过滤**（负例 embedding 相似度高于正例分位地板即丢弃）。

## 金标构建协议（无人工逐条打标的双教师确认）

1. 教师 A（判定模型，概率输出）对全部 1000 对打分；
2. 教师 B（另一模型族，生成式）独立判定；
3. **方向一致 → 确认**（约 70%）；**分歧 → 第三档位模型仲裁**；
4. 防泄漏：benchmark **永不进入训练集**；训练数据按时间分层切分。

依据：LLM 标注成本约为众包 1/30 且质量更高（Gilardi et al., PNAS 2023）；双人盲标
一致性上界本身 ~81-85%（MT-Bench），双教师一致 + 仲裁的标签质量与之同档。
完整协议与指标口径（**ECE 必须报告温度**）：[benchmark-protocol](https://github.com/MaYiding/EventTwin/blob/main/docs/benchmark-protocol.md)。

## 数据来源

30 家中国科技/汽车/互联网企业新闻（2024-01 ~ 2026-09）：4,587 篇文档 → 在线事件聚类
系统产出 10,277 条有效事件提及、约 9.5k 事件簇；事件描述为系统生成的结构化框架
（类型/主体/对象/动作/时间/证据引文）。

## 使用注意

- `gold=null` 的对（约 30%）：对比模型时按协议处理（剔除或单独报告），不要填充；
- `label` 是软概率不是二值——蒸馏训练请直接用 BCE 软目标（并做 [0.05,0.95] 裁剪，
  见训练档案的"标签饱和定律"）；
- 内容含真实新闻摘录，版权归原发布方；**限非商业研究用途**（CC BY-NC 4.0）；
- 本数据集衡量"同一事件"而非"主题相关"——用它评通用 embedding 会得到低分，这是特性不是缺陷。

## 关联

- 模型：[MaYiding/EventTwin](https://huggingface.co/MaYiding/EventTwin)（v1.1 = main，v1.0 = tag）
- 代码与档案：[GitHub - MaYiding/EventTwin](https://github.com/MaYiding/EventTwin)
- 版本：v1.2 = benchmark `bm_v1_dual`（不变）+ 训练配方 v14（train-pairs-52k，v1.3/v1.4
  的数据族）；v1.1 = 配方 v9（train-pairs-25k，tag `v1.1`）；v1.0 = 配方 v8
  （train-pairs-24k，tag `v1.0`）。映射见
  [VERSIONS](https://github.com/MaYiding/EventTwin/blob/main/VERSIONS.md)

## Citation

```bibtex
@misc{eventtwin-data-2026,
  title  = {EventTwin-Data: Soft-Labeled Chinese Event Pairs for Same-Event Judgment},
  author = {Ma, Yiding},
  year   = {2026},
  url    = {https://huggingface.co/datasets/MaYiding/EventTwin-Data},
  note   = {1k dual-teacher benchmark + 24k soft-labeled training pairs}
}
```
