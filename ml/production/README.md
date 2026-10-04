# 同事件判别器 · 生产部署包

## 模型组成

| 模型 | 底座 | 参数量 | 权重 | 角色 |
|---|---|---|---|---|
| v15 | Qwen3-Reranker-4B | 4B | `models/v15_4b_merged/` | 主力判别（gray 层强） |
| v8 | bge-reranker-v2-m3 | 568M | `models/v8/` | pos 层增强 |
| v10a | bge-reranker-v2-m3 | 568M | `models/v10a/` | pos 层增强 |

## 集成公式
```
final_score = sigmoid( 0.8 × logit(v15/T15) + 0.1 × logit(v8/T8) + 0.1 × logit(v10a/T10a) ) / T_ens
```
其中各模型温度校准值存在各目录的 `calibration.json`，集成温度 T_ens = 0.5

## 部署

### 方式一：Python 直接调用
```python
from ml.judges import EnsembleJudge
judge = EnsembleJudge(
    models={"v15": "models/v15_4b_merged", "v8": "models/v8", "v10a": "models/v10a"},
    weights={"v15": 0.8, "v8": 0.1, "v10a": 0.1},
    temperature=0.5,
    tta=True,  # 对称性 TTA（score(A,B) 与 score(B,A) 平均）
)
scores = judge.judge_batch([({"frame": "事件A", "type": "price_change"},
                              {"frame": "事件B", "type": "price_change"})])
```

### 方式二：启动推理服务
```bash
python3 -m ml.production.server --port 8601
# POST /judge {"events_a": [...], "events_b": [...]}
```

## 性能指标（benchmark 1000 对）
- AUROC: 0.981
- gray 层 AUROC: 0.979
- pos 层 AUROC: 0.922
- ECE: 0.096
- 与 Jev 教师一致率: >0.92
- 推理延迟: ~15ms/对 (GPU) / ~150ms/对 (CPU)
