# -*- coding: utf-8 -*-
"""L1 训练脚本：Qwen3-Embedding-8B LoRA 对比微调（同事件表示适配）。

在 GPU 机上运行（本机仅 --dry-run 校验数据）：
  python3 -m ml.train_l1_embed --dry-run                 # 校验数据与配置（无 GPU 依赖）
  python3 -m ml.train_l1_embed                           # 单卡 A100 40G+，3-8h
  python3 -m ml.train_l1_embed --base Qwen/Qwen3-Embedding-4B --epochs 3

⚠️ 底座与路径定稿（2026-09-30，方案 §8）：
- 主路：Qwen3-Embedding-4B 全参 ZeRO-3（ms-swift 官方路径，4 卡），本脚本为 ST 备选路线；
- 红线：sentence-transformers 默认 mean pooling 会毁掉 Qwen3-Embedding（HF discussions/15）
  ——用本脚本前必须确认 pooling=last_token + padding_side=left + query instruct 前缀
  且与推理端 vLLM 同构；官方替代命令：
    ms-swift sft --task_type embedding --model_type qwen3_emb \
      --model Qwen/Qwen3-Embedding-4B --train_type full --loss_type infonce \
      --dataloader_drop_last true（环境变量 INFONCE_USE_BATCH=True,
      INFONCE_MASK_FAKE_NEGATIVE=True, INFONCE_TEMPERATURE=0.01）
- LoRA r=64/α=128 与全参差 0.2pp（sbert 官方实证；但 TMLR 2024 "LoRA Learns Less"
  显示效果优先选全参），adapter 2% 参数，部署热加载零成本；
- 负例已过 margin 假负例过滤（data_build 阶段完成）；
- 训后评测：python3 -m ml.evaluator embedding（对比 jev 基线）。
产出：ml/models/l1_adapter/（vLLM 热加载，部署侧换 adapter 即可全量重嵌）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "l1_pairs.jsonl"
OUT_DIR = Path(__file__).parent / "models"


def load_rows():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--out", default=str(OUT_DIR / "l1_adapter"))
    ap.add_argument("--dry-run", action="store_true", help="只校验数据与参数，不加载模型")
    ap.add_argument("--no-cached", action="store_true",
                    help="用普通 MultipleNegativesRankingLoss（Cached 版依赖 triton，容器无编译环境时用）")
    args = ap.parse_args()

    rows = load_rows()
    neg_counts = [len(r["negatives"]) for r in rows]
    print(f"数据: {len(rows)} 行 anchor/positive/negatives（负例均值 {sum(neg_counts)/len(rows):.1f}）")
    assert rows and all(r["negatives"] for r in rows[:10]), "负例缺失，先跑 ml.data_build"
    print(f"配置: base={args.base} LoRA r={args.lora_r} epochs={args.epochs} "
          f"batch={args.batch} lr={args.lr}")
    if args.dry_run:
        print("dry-run 通过 ✓（数据格式/参数校验完成）")
        return

    # ---- GPU 训练（以下仅在训练机执行）----
    import torch
    from peft import LoraConfig
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from datasets import Dataset

    model = SentenceTransformer(args.base)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="FEATURE_EXTRACTION")
    model.add_adapter(lora)

    import random as _rnd
    _rng = _rnd.Random(42)
    all_negs = [n for r in rows for n in r["negatives"]]
    def _pad_negs(negs):
        return (list(negs) + _rng.sample(all_negs, 4))[:4]
    ds = Dataset.from_list([
        {"anchor": r["anchor"], "positive": r["positive"],
         **{f"negative_{i}": n for i, n in enumerate(_pad_negs(r["negatives"]))}}
        for r in rows])
    col_names = ["anchor", "positive", "negative_0", "negative_1",
                 "negative_2", "negative_3"]
    if args.no_cached:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        loss = MultipleNegativesRankingLoss(model)
    else:
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.batch)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=ds.select_columns(col_names),
                                         loss=loss)
    trainer.train()
    model.save_pretrained(args.out)
    print(f"adapter 已保存 {args.out}；评测：python3 -m ml.evaluator embedding "
          f"--model {args.out}（部署侧需将 adapter 挂回基座后重嵌全库）")


if __name__ == "__main__":
    main()
