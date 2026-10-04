# ml/ —— 判定模型训练与评测框架（L1 表示适配 × L2 Jev 蒸馏）

> 2026-09-30 搭建完成。设计：**框架与验证全部用 Jev 跑通**（teacher 即基线），
> 本地模型训练完成后通过 `config.json → judge` 一键切换，调用方零改动。
> 依据：`compare/L1L2专项调研与决策报告.md`（含全部文献出处与实测数据）。

## 全流程

```
① 数据构造（免费，自动）
   python3 -m ml.data_build
   → ml/data/l1_pairs.jsonl   4656 行 anchor/positive/negatives（对比训练三元组）
   → ml/data/l2_pairs.jsonl   19848 对（软标签蒸馏，含 Jev 复核升格的 821 个正对）
   → 正 6111 / 负 13737；hard 负例经 margin 初筛 + Jev 复核（假负例升格为正对）
   → 时间切分 8:1:1（防同源泄漏）

② 金标 benchmark（1000 对分层，Jev 预标已完成）
   python3 -m ml.benchmark build    # 抽样+预标（走缓存，重跑零成本）
   python3 -m ml.benchmark export   # → benchmark/review_v1.csv 人工复核表
   （人工填 human 列后）python3 -m ml.benchmark import review_v1_filled.csv
   分层：pos 300（簇内+转载）/ neg 400（跨企业）/ gray 300（同实体同类型时间重叠，最难层）
   金标以人工为准（防教师偏差泄漏）；未复核期用 jev≥0.5 临时标签并在报告标注

③ 统一评测（任意判别器 × 金标 → AUROC/acc/ECE/分层）
   python3 -m ml.evaluator jev            # teacher 基线
   python3 -m ml.evaluator embedding      # 通用表示基线（L1 的 gap 底线）
   python3 -m ml.evaluator reranker       # 未微调 cross-encoder（L2 的零训练底线，需GPU）
   python3 -m ml.evaluator reranker --model ml/models/l2_student   # 训后学生
   python3 -m ml.evaluator cascade --band 0.35   # 级联（含升级率/一致率）

④ 训练（GPU 机执行；本机只 --dry-run）
   python3 -m ml.train_l1_embed --dry-run   # ✓ 已验证
   python3 -m ml.train_l1_embed             # Qwen3-Embedding-8B LoRA r=64，单卡 A100 3-8h
   python3 -m ml.train_l2_reranker          # bge-reranker-v2-m3 软标签蒸馏，单卡 1-2h
   产出 ml/models/l1_adapter / l2_student

⑤ 生产替换（训练达标后）
   config.json → "judge": {"provider": "cascade", "cascade_local": "reranker",
                            "cascade_model": "ml/models/l2_student", "cascade_band": 0.35}
   接入点：ml/judges.py make_judge()（resolve 判别的替换入口，接口同构零改动）

## 基线数字（2026-09-30 实测，金标=Jev 预标版，待人工复核校准）

| judge | OVERALL AUROC | acc@最优阈 | gray AUROC | gray ECE | vs Jev 一致率 |
|---|---|---|---|---|---|
| Jev（teacher） | 0.9995 | 0.991 | 1.000 | 0.102 | — |
| embedding-8b（通用） | 0.9774 | 0.934 | 0.951 | 0.563 | 0.657 |

读数：通用 embedding 排序能力尚可（AUROC 0.95+）但**校准灾难（ECE 0.56）**、
阈值漂移（0.64-0.93）、与 Jev 硬判定一致率仅 65.7%——即 L1 微调的目标：
AUROC↑ + ECE→0.15 以内 + 一致率→90%+。

## 模块清单

| 文件 | 职责 |
|---|---|
| judges.py | 判别器抽象：JevPairJudge / EmbeddingPairJudge / RerankerPairJudge / CascadeJudge；make_judge() 按 config 路由 |
| data_build.py | 训练数据构造（正对挖掘 5 源 / 负对 2 源 / margin 初筛 + Jev 假负例复核升格 / 时间切分） |
| benchmark.py | 金标构建三命令（build / export / import） |
| evaluator.py | 统一评测（AUROC 秩法 / 最优阈值 acc / ECE 10 桶 / 分层 / 与教师一致率 / 级联升级率） |
| train_l1_embed.py | L1：sentence-transformers 3.x + PEFT LoRA + CachedMNRL |
| train_l2_reranker.py | L2：软标签 BCE 蒸馏（正负 1:8 平衡采样） |
| data/ · benchmark/ · models/ | 产物目录（data/benchmark 入库，models gitignored） |


## 实施状态（2026-10-01，GPU 访问待开通）

| 项 | 状态 |
|---|---|
| M0 小改 S1-S3（blocking key/实体归一/留痕全池） | ✅ 已落地并全量重放验证中 |
| M0.5 双师复核（Jev × qwen3.8-max-0902） | 🔄 进行中 → bm_v1_dual.json（一致即金标，分歧留审） |
| 原生置信度（温度缩放校准/推理温度/reliability） | ✅ 三件套已落地 |
| 训练落地包（swift_data + train_l1_swift.sh 海选→精训） | ✅ 命令就绪，GPU 开通即跑 |
| L3 canonicalize.py | ✅ 实现并 dry-run 真跑通过（18 自动并提案 0.90-0.99） |
| L1/L2 真实训练 | ⏸ 待 GPU 访问（901/902 四卡，SSH 未授权 → 需部署公钥） |
| M3 canonicalization 放开执行 | ⏸ dry-run 提案人工抽检 ≥0.9 后 |

GPU 访问 blocker：10.10.21.215/216（901/902）SSH 均无本机公钥（root/admin/mayiding
均 Permission denied）。开通后执行：`bash ml/train_l1_swift.sh`（L1 海选→精训）→
`python3 -m ml.train_l2_reranker`（L2 蒸馏+温度校准）→ 评测对决 → config 切 cascade。
