#!/usr/bin/env python3
"""Laya 分类器 v2：客户端显式截断，每次请求整体 ≤1024 token。

Laya 服务端硬截断 1024 token（v3flat 版实测），253 选项的平面 Choice 放不下，
改为逐层下降：L1→L2→L3→L4 各一次 choice，正文截前 ~550 字（按 1 中文字≈1
token 校准，请求总长控制在 ~950 token）。结果作为 Laya 最终成绩。
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, load_taxonomy, load_testset, http_json

LAYA_URL = "http://100.92.114.89:8084/v1/systemone"
CONTENT_CAP = 550  # 中文字符（预算：~40 脚手架+~110 规则+~250 最胖层选项+正文）
RULES_SHORT = (
    "判定规则：行业按文章内容判断，不按公司主业务；多主题以主要事件为准；"
    "发布带售价属产品发布；财报业绩属财报与业绩；建厂扩产属产能与扩张；"
    "高管落马属法律与监管；第四层实体产品归硬件、系统/大模型/App归软件与服务。")


def ask(options, question, item, desc=None):
    state = {"文章": {"标题": item["title"],
                     "正文": item["content"][:CONTENT_CAP]},
             "规则": RULES_SHORT}
    # Laya 要求 choice 必须带非空 criteria：无描述时用选项名兜底
    criteria = {o: (desc or {}).get(o) or o for o in options}
    q = {"type": "choice",
         "instructions": f"按 `规则` 回答：{question}",
         "criteria": criteria}
    r = http_json(LAYA_URL, {"model": "multilingual", "state": state,
                             "questions": {"pick": q}}, timeout=300)
    a = r["answers"]["pick"]
    ch = a.get("choice")
    if ch in options:
        return ch, r["usage"].get("input_tokens")
    for o in options:  # 宽松包含
        if ch and o in ch:
            return o, r["usage"].get("input_tokens")
    return None, r["usage"].get("input_tokens")


def run_one(item, tax):
    t0 = time.time()
    toks = []
    l1, tk = ask(tax["l1"], "该文章报道的内容属于哪个行业？", item)
    toks.append(tk)
    if not l1:
        return {"id": item["id"], "pred_path": [], "error": "L1 解析失败",
                "latency_s": round(time.time() - t0, 2), "tokens": toks}
    l2s = tax["l2_tree"].get(l1, [])
    l2, tk = (ask(l2s, f"文章属于「{l1}」下的哪个细分领域？", item,
                  tax.get("l2_desc")) if l2s else (None, 0))
    toks.append(tk)
    l3, tk = ask(tax["l3_list"], "文章报道的核心事件类型是什么？", item,
                 tax.get("l3_desc"))
    toks.append(tk)
    l4, tk = (None, 0)
    if l3 and l3 in tax["l4_tree"]:
        l4, tk = ask(list(tax["l4_tree"][l3].keys()),
                     f"该「{l3}」事件进一步属于哪个细分？", item,
                     tax["l4_tree"][l3])
        toks.append(tk)
    path = [x for x in (l1, l2, l3, l4) if x]
    return {"id": item["id"], "pred_path": path,
            "latency_s": round(time.time() - t0, 2), "tokens": toks}


def main():
    tax = load_taxonomy()
    items = load_testset()
    print(f"{len(items)} texts, hierarchical descent, cap {CONTENT_CAP} chars",
          flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, res in enumerate(ex.map(lambda it: run_one(it, tax), items)):
            out.append(res)
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(items)}] {'/'.join(res['pred_path'])} "
                      f"{res['latency_s']}s tok={res['tokens']}", flush=True)
    (ROOT / "results" / "laya.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    mx = max(max(r["tokens"]) for r in out if r["tokens"])
    print(f"done: {sum(1 for r in out if r.get('pred_path'))}/{len(out)} ok, "
          f"单请求最大 input_tokens={mx}")


if __name__ == "__main__":
    main()
