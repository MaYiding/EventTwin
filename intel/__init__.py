# -*- coding: utf-8 -*-
"""企业外部情报系统（事件为中心的增量情报架构）单机可复现实现。

v3 双模型架构（Jev + Qwen3-8B，依据 2026-09-24 五模型评测结论）：
- qwen3-8b 只做生成（提及抽取 / 问答）；Jev 做全部判定（同事件判别 /
  候选同事件分布 / 检索相关性重排）；embedding / reranker / Laya 已移除。

六段主线（对应《主线与决策摘要》）：
1. Raw/Truth Ledger      —— intel.store（原文快照、断言账本、双时间选择历史）
2. 混合检索与候选召回     —— intel.nlp（BM25）+ 结构化通道 + Jev 相关性分布
3. 轻量事件聚合          —— intel.pipeline.resolve（attach/create_provisional/judge/pending）
4. 时间感知事件检索       —— intel.query_service（双时间事实查询）
5. 高价值内容按需建图     —— intel.graph_service（七类关系 + 图增删改查）
6. 带证据回答            —— intel.query_service.answer（evidence pack + 流式引用）

生产架构中的 PostgreSQL/OpenSearch/Kafka/对象存储在本单机实现中的映射：
  PostgreSQL → SQLite(WAL)，同表名同职责，权威事实源
  Kafka     → SQLite 持久化队列（同 topic 划分：intel.*.v1）
  OpenSearch → 进程内 BM25（projection_state 记录投影水位，可重建）；
               语义排序由 Jev 判定模型承担，不再维护向量索引
  对象存储   → data/state/objects 内容寻址文件库
"""
__version__ = "3.0.0"
