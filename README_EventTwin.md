# EventTwin — 中文同事件判别器

判断两个新闻事件描述是否指同一个现实发生的事件（same-event detection）。

## 模型

| 模型 | 底座 | AUROC | 链接 |
|---|---|---|---|
| **v2.2 = v24b（主力）** | Qwen3-Reranker-4B | **0.985** | [HuggingFace](https://huggingface.co/MaYiding/EventTwin) |
| v2.1 = v23b | Qwen3-Reranker-4B | 0.983 |
| v2.0 = v15 | Qwen3-Reranker-4B | 0.963 | [HuggingFace tag v2.0](https://huggingface.co/MaYiding/EventTwin/tree/v2.0) |
| v8 | bge-reranker-v2-m3 (568M) | 0.909 | [HuggingFace ensemble/v8](https://huggingface.co/MaYiding/EventTwin/tree/main/ensemble/v8) |
| v10a | bge-reranker-v2-m3 (568M) | 0.903 | [HuggingFace ensemble/v10a](https://huggingface.co/MaYiding/EventTwin/tree/main/ensemble/v10a) |

## 快速使用

```python
from ml.judges import EnsembleJudge

judge = EnsembleJudge(
    models={"v15": "...", "v8": "...", "v10a": "..."},
    weights={"v15": 0.8, "v8": 0.1, "v10a": 0.1},
    temperature=0.5, tta=True,
)
score = judge.judge_batch([({"frame": "小米YU7上市"}, {"frame": "小米发布YU7"})])
```

## 性能（1000 对 benchmark）

| 指标 | v15 单模型 | 三模型集成 | **v2.1 单模型** | Jev（教师上限） |
|---|---|---|---|---|
| AUROC | 0.963 | 0.981 | **0.983** | 0.998 |
| gray 层 | 0.974 | 0.979 | **0.982** | 1.000 |
| pos 层 | 0.845 | 0.922 | **0.926** | 0.992 |
| ECE | 0.107 | 0.096 | **0.076** | 0.062 |

v2.2（内部 v24b）：v2.1 配方 + 四象限跨粒度（各 1.5k），T=0.7789，象限 hold-out 0.99-1.0，长文并簇 acc 0.983。

## 数据
- [EventTwin-Data](https://huggingface.co/datasets/MaYiding/EventTwin-Data)：1000 对分层金标 + 55K v2.1 训练对

## 技术报告
- [十九版本完整对决报告](ml/benchmark/学生模型对决报告.md)
- 训练配方：提及式数据 + LoRA + LS + 温度校准
- 集成公式：`sigmoid(0.8×logit(v15/T) + 0.1×logit(v8/T) + 0.1×logit(v10a/T)) / 0.5`
