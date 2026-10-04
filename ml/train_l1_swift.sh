#!/usr/bin/env bash
# L1 训练落地包（GPU 机执行，四卡；依据 L1L2L3-实施方案 §8.3 两阶段打法）
#
# 前置（训练机一次性）：
#   pip install -U 'ms-swift[all]'  transformers>=4.51  sentence-transformers>=5.7
#   # 模型下载（内网 HF 镜像或 ModelScope）：
#   modelscope download --model Qwen/Qwen3-Embedding-4B --local_dir ~/models/qwen3-emb-4b
#   modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ~/models/qwen3-emb-06b
#   # 数据：把本仓 ml/data/ 同步到训练机（含 swift/ 子目录，由 python3 -m ml.swift_data 生成）
#
# 红线（§8.4）：last-token pooling 由 SWIFT qwen3_emb 任务类型保证；左 padding 同；
#   query 侧 Instruct 前缀训练推理必须同构（部署 vLLM 侧同样加）；
#   跨卡负例 + drop_last + 假负例 mask 三个开关必须开。
set -euo pipefail
cd "$(dirname "$0")/.."   # CodeV3 根

DATA=ml/data/swift
OUT=ml/models

# ── 第 1 轮·海选（半天，2 组并行各占 2 卡）──────────────────────────────
# 变量：底座（4B/0.6B）× lr —— 评估 val AUROC/ECE 择 top1-2
CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --task_type embedding --model_type qwen3_emb --model ~/models/qwen3-emb-4b \
  --train_type lora --lora_rank 64 --lora_alpha 128 \
  --dataset $DATA/l1_train.jsonl --val_dataset $DATA/l1_val.jsonl \
  --loss_type infonce --temperature 0.02 \
  --torch_dtype bfloat16 --max_length 512 \
  --per_device_train_batch_size 8 --dataloader_drop_last true \
  --learning_rate 1e-4 --num_train_epochs 2 --warmup_ratio 0.1 \
  --save_strategy epoch --eval_strategy steps --eval_steps 100 \
  --output_dir $OUT/l1_sel_4b_lora &

CUDA_VISIBLE_DEVICES=2,3 swift sft \
  --task_type embedding --model_type qwen3_emb --model ~/models/qwen3-emb-06b \
  --train_type lora --lora_rank 64 --lora_alpha 128 \
  --dataset $DATA/l1_train.jsonl --val_dataset $DATA/l1_val.jsonl \
  --loss_type infonce --temperature 0.02 \
  --torch_dtype bfloat16 --max_length 512 \
  --per_device_train_batch_size 16 --dataloader_drop_last true \
  --learning_rate 2e-4 --num_train_epochs 2 --warmup_ratio 0.1 \
  --save_strategy epoch --eval_strategy steps --eval_steps 100 \
  --output_dir $OUT/l1_sel_06b_lora
wait

# ── 第 2 轮·精训（1-2 天，4 卡 ZeRO-3 全参，海选胜者底座）───────────────
# 全参 lr 按 §8.5：6e-6（Qwen 官方）起；跨卡 in-batch 负例必须开
swift sft \
  --task_type embedding --model_type qwen3_emb --model ~/models/qwen3-emb-4b \
  --train_type full \
  --dataset $DATA/l1_train.jsonl --val_dataset $DATA/l1_val.jsonl \
  --loss_type infonce --temperature 0.02 \
  --torch_dtype bfloat16 --max_length 512 --gradient_checkpointing true \
  --per_device_train_batch_size 8 --dataloader_drop_last true \
  --deepspeed zero3 \
  --learning_rate 6e-6 --num_train_epochs 3 --warmup_ratio 0.1 \
  --save_strategy epoch --eval_strategy steps --eval_steps 200 \
  --load_best_model_at_end true \
  --output_dir $OUT/l1_full_4b

# ── 训后评测（回本机或训练机）──────────────────────────────────────────
# 部署侧：vLLM 挂载新权重/adapter（query 前缀同构！）→ 全量重嵌 →
# python3 -m ml.evaluator embedding   （门禁：AUROC≥0.99 / ECE≤0.15 / 一致率≥0.90）
