#!/usr/bin/env python3
"""qwen3-8b 分类器：逐层下降（L1→L2→L3→L4），每层一次 chat 调用。"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, load_taxonomy, load_testset, chat

CONTENT_CAP = 1200  # 每次调用给正文的字符数（n_ctx=4096）


RULES = (
    "判定规则：(1)行业按文章报道的内容判断，不按公司主业务——手机公司造车、车企发"
    "眼镜、内容平台做大模型，都按本文报道的业务归类；(2)多主题时以最主要事件为准；"
    "(3)新品发布报道中的售价信息仍属产品发布，纯粹的降价涨价销量才是价格与销量；"
    "(4)发布财报业绩属财报与业绩；(5)建厂投产扩产能属产能与扩张；(6)高管落马、职务"
    "犯罪、被调查属法律与监管，正常任免裁员才属人事与组织；(7)第四层按事件主要对象"
    "定：实体产品（含车型）归硬件，系统/大模型/App归软件与服务，品牌名含鸿蒙、智行"
    "不改变硬件属性；(8)无法归入具体领域时选其他与跨界。")


def ask(options, question, item, desc=""):
    opt_text = "、".join(
        f"{o}（{desc[o]}）" if desc and o in desc else o for o in options)
    prompt = (f"文章标题：{item['title']}\n"
              f"正文摘录：{item['content'][:CONTENT_CAP]}\n\n{RULES}\n"
              f"{question}\n选项：{opt_text}\n"
              f"只输出其中一个选项的名称，不要输出其他任何内容。")
    out = chat(prompt, max_tokens=32)
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    for opt in options:  # 精确匹配优先，再宽松包含
        if opt == out:
            return opt
    for opt in options:
        if opt in out:
            return opt
    return None


def run_one(item, tax):
    t0 = time.time()
    calls = 0
    l2desc = tax.get("l2_desc", {})
    l1 = ask(tax["l1"], "该文章报道的内容属于哪个行业？", item)
    calls += 1
    if not l1:
        return {"id": item["id"], "pred_path": [], "error": "L1 解析失败",
                "latency_s": round(time.time() - t0, 2)}
    l2s = tax["l2_tree"].get(l1, [])
    l2 = ask(l2s, f"文章属于「{l1}」下的哪个细分领域？", item,
             {k: l2desc[k] for k in l2s if k in l2desc}) if l2s else None
    calls += 1
    l3 = ask(tax["l3_list"], "文章报道的核心事件类型是什么？", item,
             tax.get("l3_desc", {}))
    calls += 1
    l4 = None
    if l3 and l3 in tax["l4_tree"]:
        l4 = ask(list(tax["l4_tree"][l3].keys()),
                 f"该「{l3}」事件进一步属于哪个细分？", item,
                 tax["l4_tree"][l3])
        calls += 1
    path = [x for x in (l1, l2, l3, l4) if x]
    return {"id": item["id"], "pred_path": path, "calls": calls,
            "latency_s": round(time.time() - t0, 2)}


def main():
    tax = load_taxonomy()
    items = load_testset()
    print(f"{len(items)} texts, hierarchical descent", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, res in enumerate(ex.map(lambda it: run_one(it, tax), items)):
            out.append(res)
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(items)}] {'/'.join(res['pred_path'])} "
                      f"{res['latency_s']}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "qwen8b.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"done: {sum(1 for r in out if r.get('pred_path'))}/{len(out)} ok")


if __name__ == "__main__":
    main()
