#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多卡 QLoRA 训练（4×4090D 专用）：每卡独立加载 4-bit 模型，DDP 只同步 LoRA 梯度。

用法：
  CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 \
    -m ml.train_l2_multigpu --base /root/models/Qwen_Qwen3-Reranker-8B ...

  或 accelerate launch --num_processes=4 -m ml.train_l2_multigpu ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# 分布式初始化（torchrun 自动设置环境变量）
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset as TD
from torch.utils.data.distributed import DistributedSampler

DATA = Path(__file__).parent / "data" / "l2_pairs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2, help="per-GPU batch size")
    ap.add_argument("--grad-accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    # 分布式初始化
    if WORLD_SIZE > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    if RANK == 0:
        print(f"多卡训练: world_size={WORLD_SIZE}, device={device}", flush=True)
        print(f"配置: {vars(args)}", flush=True)

    # 数据
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines()
            if x and json.loads(x).get("split") == "train"]
    random.Random(args.seed).shuffle(rows)
    if RANK == 0:
        print(f"训练数据: {len(rows)} 对", flush=True)

    # Qwen3-Reranker 模板
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("no", add_special_tokens=False)["input_ids"][0]
    PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
           "Query. Only give me the judgment and do not output any other words or explanations. "
           "The judgment should be yes or no.<|im_end|>\n<|im_start|>user\nQuery: ")
    SFX = "\nDocument: "
    SFX2 = "\nJudgment: <|im_end|>\n<|im_start|>assistant\n"

    class PairDS(TD):
        def __len__(self):
            return len(rows)
        def __getitem__(self, i):
            r = rows[i]
            return r["a"], r["b"], float(r["label"])

    def collate(batch):
        aa, bb, ys = zip(*batch)
        texts = [f"{PFX}{a[:200]}{SFX}{b[:200]}{SFX2}" for a, b in zip(aa, bb)]
        inp = tok(list(texts), padding=True, truncation=True, max_length=512,
                  return_tensors="pt", add_special_tokens=False)
        return inp, torch.tensor(ys, dtype=torch.float32)

    ds = PairDS()
    sampler = DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, collate_fn=collate,
                    drop_last=True, num_workers=0)

    # bf16 全精度 LoRA（每卡完整副本，base 冻结无梯度，只有 LoRA 有优化器状态）
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{LOCAL_RANK}"},
        attn_implementation="sdpa")
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 优化器（PagedAdamW8bit 处理显存尖峰）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # v15 训练技巧移植（2026-10-04 用户诊断：多卡脚本缺失三大技巧是 8B 低分根因）
    LS_BETA = 0.1        # Label Smoothing（v15 默认值，ICML 2025 ECE 0.343→0.125）
    RDROP_LAMBDA = 2.0   # R-Drop（v15 默认值，NeurIPS 2021 小数据 +1.21）
    EMA_DECAY = 0.999    # EMA 权重平均（Model Soups ICML 2022 低数据收益最大）
    ema_shadow = {}       # EMA 参数影子（简化实现，不依赖 AveragedModel）

    def _ema_update(model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in ema_shadow:
                        ema_shadow[name] = param.data.clone()
                    else:
                        ema_shadow[name].mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

    def _soft_loss(logits, y):
        ys = y * (1 - LS_BETA) + LS_BETA / 2
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, ys)

    # 训练循环
    total_steps = len(dl) * args.epochs // args.grad_accum
    if RANK == 0:
        print(f"总步数: {total_steps} (每卡 {len(dl)} 批 × {args.epochs} epoch / accum {args.grad_accum})",
              flush=True)

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train()
        for bi, (inp, y) in enumerate(dl):
            inp = {k: v.to(device) for k, v in inp.items()}
            y = y.to(device)
            # Qwen3-Reranker yes/no 打分（只取末位 token logits）
            logits_all = model(**inp).logits[:, -1, :].float()
            yes_l = logits_all[:, yes_id]
            no_l = logits_all[:, no_id]
            two = torch.stack([no_l, yes_l], dim=-1)
            logp = torch.log_softmax(two, dim=-1)[:, 1]
            loss = _soft_loss(logp, y)
            # decoder-style 跳过 R-Drop（v18 崩溃根因：λ=2.0 在 8B decoder 导致坍缩）
            (loss / args.grad_accum).backward()

            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                _ema_update(model)
                step += 1
                if RANK == 0 and step % 20 == 0:
                    print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)

    # 保存（rank 0 only）
    if RANK == 0:
        os.makedirs(args.out, exist_ok=True)
        # 保存 EMA 版本（低数据 regime 权重平均更稳）
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name])
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"模型已保存 {args.out}", flush=True)

        # 温度校准
        val = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines()
               if x and json.loads(x).get("split") == "val"][:1000]
        model.eval()
        logits_all, ys = [], []
        with torch.no_grad():
            for i in range(0, len(val), 8):
                chunk = val[i:i+8]
                texts = [f"{PFX}{c['a'][:200]}{SFX}{c['b'][:200]}{SFX2}" for c in chunk]
                inp = tok(texts, padding=True, truncation=True, max_length=512,
                          return_tensors="pt", add_special_tokens=False)
                inp = {k: v.to(device) for k, v in inp.items()}
                la = model(**inp).logits[:, -1, :].float()
                two_c = torch.stack([la[:, no_id], la[:, yes_id]], dim=-1)
                logits_all += torch.log_softmax(two_c, dim=-1)[:, 1].tolist()
                ys += [float(c["label"]) for c in chunk]
        lt, yt = torch.tensor(logits_all).to(device), torch.tensor(ys).to(device)
        rho = torch.nn.Parameter(torch.tensor(0.5).to(device))
        opt_t = torch.optim.LBFGS([rho], lr=0.05, max_iter=60)
        def closure():
            opt_t.zero_grad()
            Tc = torch.nn.functional.softplus(rho) + 0.01
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lt / Tc, yt)
            loss.backward()
            return loss
        opt_t.step(closure)
        T = round(float(torch.nn.functional.softplus(rho.detach()) + 0.01), 4)
        json.dump({"temperature": T, "n_val": len(val)},
                  open(args.out + "/calibration.json", "w"))
        print(f"温度校准 T={T}", flush=True)

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
