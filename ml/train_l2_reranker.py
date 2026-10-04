# -*- coding: utf-8 -*-
"""L2 训练脚本：bge-reranker-v2-m3 软标签蒸馏（Jev → 本地 cross-encoder 学生）。

在 GPU 机上运行（本机仅 --dry-run 校验数据）：
  python3 -m ml.train_l2_reranker --dry-run
  python3 -m ml.train_l2_reranker                # 568M 全参微调，单卡 24G+，1-2h
  python3 -m ml.train_l2_reranker --lora          # LoRA 版（更省）

底座定稿（2026-09-30，方案 §8）：
- 主路：Qwen3-Reranker-4B（decoder 式 yes/no logit → 原生 P(yes) 概率，与软标签蒸馏/
  级联分带天然匹配；SWIFT 官方 4 卡配方：MAX_POSITIVE_SAMPLES=1, MAX_NEGATIVE_SAMPLES=7）；
- 本脚本默认 base 仍为 bge-reranker-v2-m3（encoder 式对照）——换主路加
  --base Qwen/Qwen3-Reranker-4B（decoder 打分路径：chat 模板限定 yes/no，取最后
  token logits softmax）；
设计依据（L1L2 调研报告）：
- 软标签回归（label 为 Jev 概率，KD 温度 2-4）优于硬 0/1（保灰带校准）；
- MiniCheck 实证：1.4-3.6 万教师标签 + 400M cross-encoder ≈ GPT-4 判定水平；
- 按时间切分已由 data_build 保证（train split）；
- 训后评测：python3 -m ml.evaluator reranker --model <ckpt>（对决 Jev 基线）。
产出：ml/models/l2_student/。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data" / "l2_pairs.jsonl"
OUT_DIR = Path(__file__).parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pairs", type=int, default=50000)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--ls-beta", type=float, default=0.1, help="label smoothing β（v10 扫参）")
    ap.add_argument("--qlora", action="store_true",
                    help="QLoRA 4-bit 量化训练（8B+ 大模型单卡必需）")
    ap.add_argument("--margin-mse", action="store_true",
                    help="MarginMSE 蒸馏（Hofstätter ECIR 2021）：学生拟合教师分数差")
    ap.add_argument("--rdrop-lambda", type=float, default=2.0, help="R-Drop KL 系数")
    ap.add_argument("--decoder-style", action="store_true",
                    help="Qwen3-Reranker 类 decoder 式打分：chat 模板限定 yes/no，"
                         "取最后 token 的 yes/no logits（方案 §8.1 L2 主路）")
    ap.add_argument("--out", default=str(OUT_DIR / "l2_student"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    if args.margin_mse:
        # MarginMSE 模式：l1_pairs 三元组（anchor/positive/negatives），无 label/split 字段
        L1 = Path(__file__).parent / "data" / "l1_pairs.jsonl"
        data = [json.loads(x) for x in L1.read_text(encoding="utf-8").splitlines() if x]
        data = data[:args.max_pairs]
        print(f"配置: MarginMSE lr={args.lr} ep={args.epochs}", flush=True)
        print(f"MarginMSE 三元组: {len(data)}", flush=True)
        if args.dry_run:
            print("dry-run 通过 ✓")
            return
    else:
        rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line).get("split") == "train"]
        rng = random.Random(args.seed)
        # 平衡采样：正对全保留，负对下采样到正对的 8 倍（防 1:176 极度不平衡支配损失）
        pos = [r for r in rows if r["label"] >= 0.5]
        neg = [r for r in rows if r["label"] < 0.5]
        neg_take = rng.sample(neg, min(len(neg), len(pos) * 8)) if pos else neg
        data = pos + neg_take
        rng.shuffle(data)
        data = data[:args.max_pairs]
        # 标签裁剪防饱和（v4-v6 教训：簇对通道的 Jev 标签呈 0.001-0.99 极端分布，
        # 未裁剪时 BCE 将 logit 推向 ±∞，输出饱和 sigmoid≈1、温度校准梯度消失）
        data = [{**r, "label": min(max(r["label"], 0.05), 0.95)} for r in data]
        print(f"配置: LS={args.ls_beta} RDrop={args.rdrop_lambda} lr={args.lr} ep={args.epochs}", flush=True)
        print(f"数据: 正 {len(pos)} / 负 {len(neg)}（采样后 {len(neg_take)}）→ 训练 {len(data)} 对")
        if args.dry_run:
            print("dry-run 通过 ✓")
            return

    import torch
    from torch.utils.data import DataLoader, Dataset as TD
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    class PairTD(TD):
        def __init__(self, rows, decoder_style=False, margin_mse=False):
            self.rows = rows
            self.decoder_style = decoder_style
            self.margin_mse = margin_mse

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            if self.margin_mse:
                return [(r["anchor"], r["positive"], 1.0)] + \
                       [(r["anchor"], neg, 0.0) for neg in r.get("negatives", [])]
            return r["a"], r["b"], float(r["label"])

    tok = AutoTokenizer.from_pretrained(args.base)
    # decoder 式：Qwen3-Reranker 官方打分模板（yes/no 二词元概率）
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
              "Query. Only give me the judgment and do not output any other words or "
              "explanations. The judgment should be \"yes\" or \"no\".<|im_end|>\n"
              "<|im_start|>user\nQuery: ")
    suffix = "\nDocument: "
    suffix2 = "\nJudgment: <|im_end|>\n<|im_start|>assistant\n"
    yes_ids = None
    if args.decoder_style:
        yes_ids = tok("yes", add_special_tokens=False)["input_ids"]
        no_ids = tok("no", add_special_tokens=False)["input_ids"]
        assert len(yes_ids) == 1 and len(no_ids) == 1, \
            f"yes/no 非单 token: {yes_ids} {no_ids}"
    if args.qlora:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb_cfg, device_map="auto",
            attn_implementation="sdpa")
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.enable_input_require_grads()
        model.config.use_cache = False
    elif args.decoder_style:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=1, torch_dtype=torch.bfloat16)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        targets = (["q_proj", "k_proj", "v_proj"] if args.decoder_style
                   else ["query", "key", "value"])   # Qwen decoder 系 vs BERT encoder 系
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                                 target_modules=targets))
    model.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def collate(batch):
        if args.margin_mse:
            flat_a, flat_b, flat_y, flat_aid = [], [], [], []
            for items in batch:
                aid = len(set(flat_aid))
                for a, b, label in items:
                    flat_a.append(a); flat_b.append(b)
                    flat_y.append(label); flat_aid.append(aid)
            inp = tok(flat_a, flat_b, padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            return inp, (torch.tensor(flat_y, dtype=torch.float32),
                         torch.tensor(flat_aid, dtype=torch.long))
        aa, bb, ys = zip(*batch)
        if args.decoder_style:
            texts = [f"{prefix}{q}{suffix}{d}{suffix2}" for q, d in zip(aa, bb)]
            inp = tok(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt", add_special_tokens=False)
            # 左 padding（取右侧最后 token，与推理红线一致）
            tok.padding_side = "left"
            inp = tok(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt", add_special_tokens=False)
            return inp, torch.tensor(ys, dtype=torch.float32)
        inp = tok(list(aa), list(bb), padding=True, truncation=True,
                  max_length=512, return_tensors="pt")
        return inp, torch.tensor(ys, dtype=torch.float32)

    dl = DataLoader(PairTD(data, margin_mse=args.margin_mse),
                    batch_size=args.batch, shuffle=True, collate_fn=collate)
    total_steps = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.05), total_steps)
    bce = torch.nn.BCEWithLogitsLoss()
    # v9 组合拳（依据顶会调研）：LS β=0.1 保 ECE（ICML 2025）+ R-Drop KL 一致性
    # （NeurIPS 2021 小数据 +1.21）+ EMA 权重平均（Model Soups：低数据收益最大）
    LS_BETA = args.ls_beta
    RDROP_LAMBDA = args.rdrop_lambda
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda avg, new, n:
                                              0.999 * avg + 0.001 * new)

    def _soft_loss(logits, y):
        ys = y * (1 - LS_BETA) + LS_BETA / 2     # 软标签 + label smoothing
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, ys)

    step = 0
    for ep in range(args.epochs):
        model.train()
        for inp, y_data in dl:
            _dev = next(model.parameters()).device
            inp = {k: v.to(_dev) for k, v in inp.items()}
            if args.margin_mse:
                # MarginMSE（Hofstätter ECIR 2021）：学生 margin 拟合教师 margin，raw logit
                y, anchor_ids = y_data
                y = y.to(_dev); anchor_ids = anchor_ids.to(_dev)
                logits = model(**inp).logits.squeeze(-1).float()
                pos_mask = y == 1.0
                neg_mask = ~pos_mask
                total_loss = torch.tensor(0.0, device=logits.device)
                n_pairs = 0
                for aid in anchor_ids.unique():
                    a_mask = anchor_ids == aid
                    a_pos = logits[a_mask & pos_mask]
                    a_neg = logits[a_mask & neg_mask]
                    if len(a_pos) > 0 and len(a_neg) > 0:
                        # 教师 margin：正例标签 1.0 - 负例标签 0.0 → logit 尺度估计 2.0
                        # （Jev 软标签场景下取 soft 标签差；二元代理下恒定 2.0）
                        t_margin = (a_pos.mean() * 0 + 2.0)  # 常量教师 margin
                        s_margin = a_pos.mean() - a_neg.mean()
                        total_loss = total_loss + torch.nn.functional.mse_loss(
                            s_margin, t_margin)
                        n_pairs += 1
                if n_pairs == 0:
                    continue
                loss = total_loss / n_pairs
            else:
                y = y_data.to(_dev)
                if args.decoder_style and yes_ids is not None:
                    # Qwen3-Reranker decoder 式：取末位 token 的 yes/no 两 logit
                    logits_all = model(**inp).logits[:, -1, :].float()
                    yes_l = logits_all[:, yes_ids[0]]
                    no_l = logits_all[:, no_ids[0]]
                    two = torch.stack([no_l, yes_l], dim=-1)
                    logp = torch.log_softmax(two, dim=-1)[:, 1]
                    loss = _soft_loss(logp, y)
                else:
                    logits = model(**inp).logits.squeeze(-1).float()
                    loss = _soft_loss(logits, y)
                    # R-Drop：第二次 dropout 前向 + 对称 KL（margin-mse/decoder 跳过）
                    logits2 = model(**inp).logits.squeeze(-1).float()
                    p1, p2 = torch.sigmoid(logits), torch.sigmoid(logits2)
                    kl = (p1 * (torch.log(p1 + 1e-7) - torch.log(p2 + 1e-7))
                          + (1 - p1) * (torch.log(1 - p1 + 1e-7) - torch.log(1 - p2 + 1e-7)))
                    loss = loss + RDROP_LAMBDA * kl.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            ema.update_parameters(model)
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f}", flush=True)
            if step % 500 == 0:   # 步级 checkpoint（容器环境曾两次无声杀死尾段进程）
                model.save_pretrained(args.out + "_ckpt")
                tok.save_pretrained(args.out + "_ckpt")
    # ---- 温度缩放后校准（原生置信度的关键收尾，Guo et al. 2017 标准做法）----
    import os as _os
    _os.makedirs(args.out, exist_ok=True)
    # 分类头 sigmoid 天然过自信；软标签直传 Jev 校准概率后，再在 val split 上
    # 拟合单一温度 T 最小化 NLL，输出校准配置随 checkpoint 一起保存——
    # 推理侧 logits/T 后过 sigmoid，ECE 才能进 0.15 以内（评测器已按此读取）。
    val = [json.loads(x) for x in
           (Path(__file__).parent / "data" / "l2_pairs.jsonl").read_text(encoding="utf-8").splitlines()
           if x and json.loads(x).get("split") == "val"]
    if val:
        model.eval()
        logits_all, ys = [], []
        with torch.no_grad():
            for i in range(0, len(val), args.batch):
                chunk = val[i:i + args.batch]
                if args.decoder_style:
                    tok.padding_side = "left"
                    texts = [f"{prefix}{c['a']}{suffix}{c['b']}{suffix2}" for c in chunk]
                    inp = tok(texts, padding=True, truncation=True, max_length=512,
                              return_tensors="pt", add_special_tokens=False)
                else:
                    inp = tok([c["a"] for c in chunk], [c["b"] for c in chunk],
                              padding=True, truncation=True, max_length=512, return_tensors="pt")
                _dev = next(model.parameters()).device
                inp = {k: v.to(_dev) for k, v in inp.items()}
                if args.decoder_style:
                    la = model(**inp, num_logits_to_keep=1).logits[:, -1, :].float()
                    logits_all += torch.log_softmax(
                        torch.stack([la[:, no_ids[0]], la[:, yes_ids[0]]], dim=-1), dim=-1)[:, 1].tolist()
                else:
                    logits_all += model(**inp).logits.squeeze(-1).float().tolist()
                ys += [float(c["label"]) for c in chunk]
        logits_t = torch.tensor(logits_all).cuda()
        ys_t = torch.tensor(ys).cuda()
        # softplus 参数化保证 T>0（v10 实战：无约束 LBFGS 发散到负温度翻转 sigmoid）
        rho = torch.nn.Parameter(torch.tensor(0.5).cuda())
        opt_t = torch.optim.LBFGS([rho], lr=0.05, max_iter=60)

        def closure():
            opt_t.zero_grad()
            Tc = torch.nn.functional.softplus(rho) + 0.01
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_t / Tc, ys_t)
            loss.backward()
            return loss
        opt_t.step(closure)
        t_val = round(float(torch.nn.functional.softplus(rho.detach()) + 0.01), 4)
        json.dump({"temperature": t_val, "n_val": len(val)},
                  open(Path(args.out) / "calibration.json", "w"))
        print(f"温度缩放: T={t_val}（val n={len(val)}，已存 calibration.json）")

    ema.module.save_pretrained(args.out)   # EMA 权重平均版（低数据 regime 更稳）
    tok.save_pretrained(args.out)
    print(f"学生已保存（EMA） {args.out}；评测：python3 -m ml.evaluator reranker --model {args.out}")


if __name__ == "__main__":
    main()
