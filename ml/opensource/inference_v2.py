#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventTwin v2.0 推理示例（Qwen3-Reranker-4B decoder 式）。

权重：https://huggingface.co/MaYiding/EventTwin（main = v2.0；v1.x 为 tag，568M encoder
版用 inference.py）

★ v2.0 三条协议红线（缺一性能大幅劣化）：
1) 每侧文本截断 200 字符（全文输入 AUROC 0.834 → 截断 0.963）；
2) 左 padding + 取序列末位（-1）的 yes/no 双 logit；
3) softmax 前除温度 T=0.6856（calibration.json 随权重发布）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "MaYiding/EventTwin"

PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
       "Query. Only give me the judgment and do not output any other words or explanations. "
       'The judgment should be "yes" or "no".<|im_end|>\n<|im_start|>user\nQuery: ')
SFX, SFX2 = "\nDocument: ", "\nJudgment: <|im_end|>\n<|im_start|>assistant\n"


def load(model_id: str = MODEL_ID, local_files_only: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    tok.padding_side = "left"                                    # 红线 2
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only).cuda().eval()
    cal = Path(model_id) / "calibration.json"
    T = json.loads(cal.read_text())["temperature"] if cal.exists() else 0.6856
    yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("no", add_special_tokens=False)["input_ids"][0]
    return tok, model, T, yes_id, no_id


def same_event_prob(a: str, b: str, *, tok, model, T, yes_id, no_id) -> float:
    a, b = a[:200], b[:200]                                     # 红线 1
    inp = tok(f"{PFX}{a}{SFX}{b}{SFX2}", return_tensors="pt", truncation=True,
              max_length=512, add_special_tokens=False)
    with torch.no_grad():
        la = model(**inp.to("cuda")).logits[:, -1, :].float()    # 红线 2
    two = torch.stack([la[:, no_id], la[:, yes_id]], dim=-1)
    return float(torch.softmax(two / T, dim=-1)[0, 1])           # 红线 3


def same_event_prob_batch(pairs: list[tuple[str, str]], *, tok, model, T, yes_id, no_id,
                          batch: int = 32) -> list[float]:
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            texts = [f"{PFX}{a[:200]}{SFX}{b[:200]}{SFX2}" for a, b in chunk]
            inp = tok(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt", add_special_tokens=False).to("cuda")
            la = model(**inp).logits[:, -1, :].float()
            two = torch.stack([la[:, no_id], la[:, yes_id]], dim=-1)
            out += [float(x) for x in torch.softmax(two / T, dim=-1)[:, 1]]
    return out


def route(p: float, auto: float = 0.9, reject: float = 0.1) -> str:
    """级联路由（v2.0 建议）：高置信直判；~3% 饱和误并核 → 误并零容忍场景配 v1.2 前置。"""
    if p >= auto:
        return "SAME（自动归并；零容忍场景建议 v1.2 前置复核）"
    if p <= reject:
        return "DIFF（自动拒并）"
    return "GRAY（升级终审/人工）"


if __name__ == "__main__":
    tok, model, T, yes_id, no_id = load()
    print(f"T = {T}\n")
    print(same_event_prob("小米YU7正式上市，售价25.35万元起，共推出三款车型",
                          "小米发布YU7系列 售价25.35万起",
                          tok=tok, model=model, T=T, yes_id=yes_id, no_id=no_id))
    print(same_event_prob("小米YU7正式上市，售价25.35万元起",
                          "特斯拉Model 3 全系降价1.5万元",
                          tok=tok, model=model, T=T, yes_id=yes_id, no_id=no_id))
    pairs = [("京东发布2024年11.11战报", "京东公布双十一购物用户数同比增长"),
             ("蔚来ES8交付创新高", "小鹏G6交付创新高")]
    for (a, b), p in zip(pairs, same_event_prob_batch(
            pairs, tok=tok, model=model, T=T, yes_id=yes_id, no_id=no_id)):
        print(f"{p:.3f}  {route(p):<24s} {a[:16]} ⊕ {b[:16]}")
