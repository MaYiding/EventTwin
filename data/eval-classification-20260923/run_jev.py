#!/usr/bin/env python3
"""Jev 分类器：flat Choice，180 个叶子一次选。"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, load_taxonomy, load_testset, leaf_desc, or_decisions

CONTENT_CAP = 2500  # 字符


def build_body(item, tax, leaf_docs):
    criteria = {lf["id"]: leaf_docs[lf["id"]] for lf in tax["leaves"]}
    return {
        "model": "typesafe/jev-1.13-20260917",
        "state": {"文章": {
            "标题": item["title"],
            "正文": item["content"][:CONTENT_CAP],
            "来源": item["source"]}},
        "questions": {"classify": {
            "type": "choice",
            "instructions": (
                "把 `文章` 归档到最合适的一个叶子类别。判定规则："
                "(1) 行业按文章报道的内容判断，不按公司主业务——手机公司造车、"
                "车企发布眼镜、内容平台做大模型，都按本文实际报道的业务归档；"
                "(2) 多主题时以最主要事件为准，次要信息不改变归类；"
                "(3) 新品发布报道中公布的售价信息仍属产品发布，纯粹的降价/涨价/"
                "销量数据报道才是价格与销量；"
                "(4) 发布财报、业绩报告属财报与业绩，不是产品发布；"
                "(5) 建厂、投产、扩产能（含投资某地建厂）属产能与扩张，不属资本与并购；"
                "(6) 高管落马、职务犯罪、被调查属法律与监管，正常任免、离职、裁员"
                "才属人事与组织；"
                "(7) 第四层按事件主要对象定：实体产品（含车型）归硬件，系统/大模型/"
                "App 归软件与服务，品牌名含鸿蒙、智行不改变硬件属性；"
                "(8) 确实无法归入具体领域时，选其他与跨界。"),
            "criteria": criteria}},
    }


def run_one(args):
    item, tax, leaf_docs = args
    t0 = time.time()
    try:
        r = or_decisions(build_body(item, tax, leaf_docs))
        a = r["answers"]["classify"]
        probs = a.get("probabilities", {})
        top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3] if probs else []
        return {"id": item["id"], "choice": a.get("choice"),
                "confidence": a.get("confidence"),
                "top3": [k for k, _ in top3], "top3_p": [round(v, 4) for _, v in top3],
                "latency_s": round(time.time() - t0, 2),
                "cost": r.get("usage", {}).get("cost"),
                "usage": r.get("usage")}
    except Exception as e:  # noqa: BLE001
        return {"id": item["id"], "error": str(e)[:300],
                "latency_s": round(time.time() - t0, 2)}


def main():
    tax = load_taxonomy()
    items = load_testset()
    leaf_docs = {lf["id"]: leaf_desc(tax, lf) for lf in tax["leaves"]}
    print(f"{len(items)} texts, {len(leaf_docs)} leaves", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(
                run_one, [(it, tax, leaf_docs) for it in items])):
            out.append(res)
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(items)}] last={res.get('choice')} "
                      f"{res['latency_s']}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "jev.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    ok = [r for r in out if "choice" in r]
    print(f"done: {len(ok)}/{len(out)} ok")


if __name__ == "__main__":
    main()
