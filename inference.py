#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event Same-Judge ZH 推理示例：单对 / 批量 / 结构化模板输入。

权重：https://huggingface.co/MaYiding/EventTwin
要点：
1) 输出是"同一现实事件"的概率（0-1）；
2) ★ 必须先除温度 T 再 sigmoid——温度随权重目录的 calibration.json 发布（v8=0.824），
   不除温度的 sigmoid 是过自信原始分，不能用于置信分带；
3) 结构化输入（框架+[类型]+（时间）+证据）与训练同构，效果最佳，但纯文本对也可用。

CPU 部署提示：transformers 直接跑约 30-80ms/对；生产建议导出 ONNX int8
（optimum-cli export onnx --model <ckpt> --framework pt out_dir）后吞吐再翻 2-3 倍。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MaYiding/EventTwin"


def load(model_id: str = MODEL_ID, local_files_only: bool = False):
    """加载模型与校准温度（calibration.json 优先，缺省回退 0.824）。"""
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, local_files_only=local_files_only).eval()
    cal_path = Path(model_id) / "calibration.json"
    T = json.loads(cal_path.read_text())["temperature"] if cal_path.exists() else 0.824
    return tok, model, T


def same_event_prob(a: str, b: str, *, tok, model, T: float) -> float:
    """单对判定：返回 P(同一现实事件)，已温度校准。"""
    inp = tok(a, b, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logit = model(**inp).logits.squeeze(-1)
    return torch.sigmoid(logit / T).item()


def same_event_prob_batch(pairs: list[tuple[str, str]], *, tok, model, T: float,
                          batch: int = 32) -> list[float]:
    """批量判定（GPU 吞吐约 200-400 对/秒 @4090）。"""
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            inp = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                      truncation=True, max_length=512, return_tensors="pt")
            logits = model(**inp).logits.squeeze(-1)
            out += [float(x) for x in torch.sigmoid(logits / T)]
    return out


def frame(text: str, type_: str | None = None, time_: str | None = None,
          evidence: str | None = None) -> str:
    """结构化输入模板（与训练同构，推荐）：框架 + [类型] + (时间) + 证据。"""
    parts = [text]
    if type_:
        parts.append(f"[{type_}]")
    if time_:
        parts.append(f"({time_})")
    if evidence:
        parts.append(f"证据“{evidence[:80]}”")
    return " ".join(p for p in parts if p)


def route(p: float, auto: float = 0.9, reject: float = 0.1) -> str:
    """置信分带路由示例（生产级联设计）：高置信直判，灰带升级强判定器/人工。"""
    if p >= auto:
        return "SAME（自动归并）"
    if p <= reject:
        return "DIFF（自动拒并）"
    return "GRAY（升级终审/人工）"


if __name__ == "__main__":
    tok, model, T = load()
    print(f"温度 T = {T}\n")

    # 1) 纯文本对
    print("—— 纯文本 ——")
    print(same_event_prob("小米YU7正式上市，售价25.35万元起，共推出三款车型",
                          "小米发布YU7系列 售价25.35万起", tok=tok, model=model, T=T))   # ~0.93
    print(same_event_prob("小米YU7正式上市，售价25.35万元起",
                          "特斯拉Model 3 全系降价1.5万元", tok=tok, model=model, T=T))   # ~0.02

    # 2) 结构化输入（训练同构，最佳效果）
    print("\n—— 结构化 ——")
    a = frame("小米YU7正式上市 售价25.35万起", "product_launch", "2025-06-26~2025-06-27")
    b = frame("小米发布YU7 售价25.35万元起", "product_launch", "2025-06-26")
    print(same_event_prob(a, b, tok=tok, model=model, T=T))

    # 3) 批量 + 路由
    print("\n—— 批量路由 ——")
    pairs = [(a, b),
             (frame("京东发布2024年11.11战报", "sales_report", "2024-11-11"),
              frame("京东公布双十一购物用户数同比增长", "sales_report", "2024-11-11")),
             (frame("蔚来ES8交付", "sales_report", "2025-01"),
              frame("小鹏G6交付", "sales_report", "2025-01"))]
    for (pa, pb), p in zip(pairs, same_event_prob_batch(pairs, tok=tok, model=model, T=T)):
        print(f"{p:.3f}  {route(p):<14s} {pa[:24]} ⊕ {pb[:24]}")
