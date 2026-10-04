#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四方对决：Jev vs qwen3.8-27b vs qwen3.7-plus vs qwen3.8-max（同事件判定）。

目的：验证"高级模型打标替代人工"的可行性——
  1) 各模型与 Jev 的一致率（整体/分层）；
  2) Jev 判定失败锚点（pos 层 Jev<0.5 的样本）上千问各档的挽回率；
  3) 分歧率 → 高级模型预标 + 仅分歧人工复核能省多少人力；
  4) 多数投票伪金标下各家的准确率。
产物：ml/benchmark/showdown.json + 控制台报告。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intel import llm  # noqa: E402

BM = json.loads((Path(__file__).parent / "benchmark" / "bm_v1.json").read_text(encoding="utf-8"))
OUT = Path(__file__).parent / "benchmark" / "showdown.json"
_STATE: dict = {}

MODELS = {
    "qwen3.8-27b": {"id": "qwen/qwen3.8-27b", "kw": {}},
    "qwen3.7-plus": {"id": "qwen/qwen3.7-plus", "kw": {"reasoning": {"enabled": False}}},
    "qwen3.8-max": {"id": "qwen/qwen3.8-max-0902", "kw": {}},
}

RULES = ("同事件判定规则：原子事件=特定参与方+特定对象+特定时间+一次具体动作；同一动作被不同媒体"
         "报道（含转载、中英文）=同一事件；同一产品先后两次不同调价/发布=不同事件；并购的宣布与"
         "交割=不同事件；旧闻被重新报道=同一事件（按事件发生时间不是报道时间）；官方更正此前报道"
         "的金额/日期=同一事件；主题相近但动作、对象或时间不同=不同事件。只按给出的信息判断，"
         "不按常识补充。")


def ask(model: str, a: str, b: str, retries: int = 3) -> float | None:
    llm._load_env()
    key = os.environ["OR_KEY"]
    spec = MODELS[model]
    prompt = (f"{RULES}\n\n事件甲：{a}\n\n事件乙：{b}\n\n"
              "判断两个事件是否为同一个现实发生的原子事件。"
              '只输出 JSON：{"same_event": 0到100的整数}，100=肯定同一事件，0=肯定不是。/no_think')
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": spec["id"], "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 2500, "temperature": 0.0, **spec["kw"]},
                timeout=240)
            r.raise_for_status()
            m = r.json()["choices"][0]["message"]
            c = m.get("content") or ""
            if not c and m.get("reasoning"):
                c = m["reasoning"]  # 极少数 content 空、答案落在 reasoning 尾部
            mm = re.search(r"\{[^{}]*same_event[^{}]*\}", c, re.S)
            if mm:
                v = json.loads(mm.group(0))["same_event"]
                return max(0.0, min(1.0, float(v) / 100.0))
            mm = re.search(r"same_event[\"'：:\s]*(\d{1,3})", c)
            if mm:
                return max(0.0, min(1.0, int(mm.group(1)) / 100.0))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  [{model}] 解析失败: {str(e)[:80]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def run_model(model: str, todo_idx: list[int], scores: list) -> list:
    import threading as _th
    lock = _th.Lock()
    done = [0]

    def _save():
        OUT.write_text(json.dumps(_STATE, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(ask, model, BM["pairs"][i]["a"], BM["pairs"][i]["b"]): i
                for i in todo_idx}
        for f in as_completed(futs):
            i = futs[f]
            try:
                scores[i] = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [{model}] #{i} 异常: {str(e)[:60]}", flush=True)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0:
                    _save()
                    print(f"  [{model}] {done[0]}/{len(todo_idx)}", flush=True)
    _save()
    ok = sum(1 for s in scores if s is not None)
    print(f"[{model}] 完成 {ok}/{len(todo_idx)}", flush=True)
    return scores


def agreement(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return None
    return round(sum(1 for x, y in pairs if (x >= 0.5) == (y >= 0.5)) / len(pairs), 4), len(pairs)


def _subset_idx():
    """关键子集：pos 全部（含 Jev 判非锚点 106）+ neg 抽 100 + gray 抽 200。"""
    import random as _rd
    rng = _rd.Random(20260930)
    idx = [i for i, l in enumerate(BM["pairs"] if False else [p["layer"] for p in BM["pairs"]])
           if l == "pos"]
    neg = [i for i, p in enumerate(BM["pairs"]) if p["layer"] == "neg"]
    gray = [i for i, p in enumerate(BM["pairs"]) if p["layer"] == "gray"]
    idx += rng.sample(neg, min(100, len(neg))) + rng.sample(gray, min(200, len(gray)))
    return sorted(set(idx))


def main():
    global _STATE
    if OUT.exists():   # 断点续跑
        _STATE = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"续跑：从 {OUT} 恢复", flush=True)
    else:
        _STATE = {"pairs": len(BM["pairs"]),
                  "jev": [p["jev_score"] for p in BM["pairs"]],
                  "layers": [p["layer"] for p in BM["pairs"]]}
    subset = _subset_idx()
    print(f"子集 {len(subset)}/{len(BM['pairs'])} 对（pos300+neg100+gray200）", flush=True)
    for model in MODELS:
        if _STATE.get(model) and all(s is not None for i, s in zip(subset, [_STATE[model][i] for i in subset])):
            print(f"== {model} 已完成，跳过 ==", flush=True)
            continue
        print(f"== 跑 {model} ==", flush=True)
        _STATE.setdefault(model, [None] * len(BM["pairs"]))
        todo = [i for i in subset if _STATE[model][i] is None]
        t0 = time.time()
        run_model(model, todo, _STATE[model])
        print(f"  耗时 {time.time()-t0:.0f}s", flush=True)
    analyze(_STATE)


def analyze(out):
    jev = out["jev"]
    layers = out["layers"]
    print("\n======== 分析 ========")
    names = list(MODELS)
    # 1) 与 Jev 的一致率（整体/分层）
    print("\n-- 与 Jev 硬判定一致率 --")
    for m in names:
        print(f"{m}: overall={agreement(jev, out[m])}")
        for layer in ("pos", "neg", "gray"):
            idx = [i for i, l in enumerate(layers) if l == layer]
            a = agreement([jev[i] for i in idx], [out[m][i] for i in idx])
            print(f"    {layer}: {a}")
    # 2) Jev 失败锚点：pos 层 Jev<0.5（与簇结构分歧的样本）
    print("\n-- Jev 判非锚点（pos 层 Jev<0.5，n=%d）：千问挽回率 --" %
          sum(1 for i, l in enumerate(layers) if l == "pos" and jev[i] < 0.5))
    idx = [i for i, l in enumerate(layers) if l == "pos" and jev[i] < 0.5]
    for m in names:
        rev = [1 for i in idx if out[m][i] is not None and out[m][i] >= 0.5]
        print(f"{m}: 判同 {len(rev)}/{len(idx)} ({len(rev)/len(idx):.0%})")
    # 反向：neg 层 Jev 判同=0（Jev 全对），千问有没有乱放
    print("\n-- neg 层（应为异事件）误判率 --")
    idx = [i for i, l in enumerate(layers) if l == "neg"]
    for m in names + ["jev"]:
        wrong = [1 for i in idx if out[m][i] is not None and out[m][i] >= 0.5]
        print(f"{m}: 误判同 {len(wrong)}/{len(idx)} ({len(wrong)/len(idx):.0%})")
    # 3) 多数投票伪金标下各家与投票的一致性（4 方投票，含 Jev）
    print("\n-- 4 方多数投票（≥3 票）为伪金标：各方一致率 --")
    votes = []
    for i in range(out["pairs"]):
        vs = [1 if jev[i] >= 0.5 else 0] + [1 if out[m][i] is not None and out[m][i] >= 0.5 else 0
                                             for m in names]
        votes.append(vs)
    for k, name in enumerate(["jev"] + names):
        agree = sum(1 for i, vs in enumerate(votes) if vs[k] == (1 if sum(vs) >= 3 else 0))
        print(f"{name}: {agree}/{len(votes)} ({agree/len(votes):.1%})")
    # 4) 全一致率（可直接采信、免人工的比例）
    all4_same = sum(1 for vs in votes if sum(vs) in (0, 4))
    split43 = sum(1 for vs in votes if sum(vs) in (1, 3))
    print(f"\n4方全一致: {all4_same}/{len(votes)} ({all4_same/len(votes):.0%})"
          f" | 3:1 分歧: {split43} ({split43/len(votes):.0%})"
          f" | 2:2 僵持: {len(votes)-all4_same-split43}")
    # 分层分歧例（展示 3 个 pos 层 Jev 判非的样本全模型打分）
    print("\n-- 分歧样例（pos 层，Jev 判非）--")
    shown = 0
    for i, l in enumerate(layers):
        if l != "pos" or jev[i] >= 0.5 or shown >= 3:
            continue
        p = BM["pairs"][i]
        print(f"[{i}] jev={jev[i]:.2f} " +
              " ".join(f"{m}={out[m][i]:.2f}" if out[m][i] is not None else f"{m}=NA" for m in names))
        print(f"   甲: {p['a'][:70]}")
        print(f"   乙: {p['b'][:70]}")
        shown += 1


if __name__ == "__main__":
    main()
