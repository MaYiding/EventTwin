#!/usr/bin/env python3
"""v2：实体(29)/事件(40)任务，四模型全跑（Jev/8B/rerank/embedding）。

v2 改进：
- 实体扩到 29 个：补 Venmo/CBS News/San Jose/Apple Intelligence/iPhone 15 Pro
  （真值在文），Tim Cook（知识陷阱：Apple CEO 但文中未出现）、iPhone 17（同系列
  新一代陷阱，文中只有 iPhone 16）。
- 事件 c01 族时间改为"今年早些时候宣布"（去掉推断的年份，消除 v1 的标注过严）。
- 判定规则写进 Jev 的 state 与 8B 的 prompt（拼写相近≠同一实体、系列换代不算、
  相对时间表述一致、金额上限≠固定值等）。
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, chat, embed, cosine, or_decisions, rerank

sys.path.insert(0, str(ROOT))

# ---------------- 实体 v2（29）----------------
ENTITIES_V2 = [  # (key, 名称, 组, 类型)
    ("apple", "Apple", "A", "Company"),
    ("apple_inc", "Apple Inc.", "A", "Company"),
    ("iphone", "iPhone", "A", "Product"),
    ("iphone_15_pro", "iPhone 15 Pro", "A", "Product"),
    ("siri", "Siri", "A", "Product"),
    ("apple_intelligence", "Apple Intelligence", "A", "Product"),
    ("ryan_clarkson", "Ryan Clarkson", "A", "Person"),
    ("clarkson_law_firm", "Clarkson Law Firm", "A", "Organization"),
    ("cbs_news", "CBS News", "A", "Organization"),
    ("paypal", "PayPal", "A", "Company"),
    ("venmo", "Venmo", "A", "Company"),
    ("san_jose", "San Jose", "A", "Location"),
    ("landsheft", "Landsheft", "A", "Person"),
    ("samsung", "Samsung", "B", "Company"),
    ("google", "Google", "B", "Company"),
    ("elon_musk", "Elon Musk", "B", "Person"),
    ("tim_cook", "Tim Cook", "B", "Person"),  # 知识陷阱：Apple CEO，文中未出现
    ("microsoft", "Microsoft", "B", "Company"),
    ("amazon", "Amazon", "B", "Company"),
    ("ipad", "iPad", "C", "Product"),
    ("iphone_17", "iPhone 17", "C", "Product"),  # 系列换代陷阱
    ("android", "Android", "C", "Product"),
    ("ftc", "Federal Trade Commission", "C", "Organization"),
    ("clarson_law_firm", "Clarson Law Firm", "D", "Organization"),
    ("appel_inc", "Appel Inc.", "D", "Company"),
    ("ryan_clark", "Ryan Clark", "D", "Person"),
    ("clarkson_university", "Clarkson University", "D", "Organization"),
    ("venmon", "Venmon", "D", "Company"),
    ("cupertin", "Cupertin", "D", "Location"),  # Cupertino 拼写伪造
]

ENTITY_RULES = (
    "实体出现判定规则：(1) 全称、简称、常见别名、明确的代词指代都算出现；"
    "(2) 字面相近但指不同对象的不算（如 Clarkson University ≠ 文中的 Clarkson "
    "Law Firm）；(3) 拼写有出入的编造名称不算（Clarson ≠ Clarkson）；"
    "(4) 实体与文章主题相关、或与文中实体同属一家公司/系列，不等于该实体本身出现"
    "（如文中报道 Apple 不等于 Tim Cook 出现，文中的 iPhone 16 不等于 iPhone 17 "
    "出现）；(5) 按文章实际文本判断，不按常识补充。")

EVENT_RULES = (
    "事件核对判定规则：(1) 金额、数量、日期与文章不符（无论差距多小、方向如何）即"
    "不一致；(2) 「最高可达/最多」与「固定值」不等价；(3) 文章未给出时间而声明给出"
    "具体时间判不一致；文章只给相对时间（如今年早些时候）而声明同样以相对时间表述"
    "且不矛盾，判一致；(4) 等价改写判一致：日期书写格式不同、同义表述、全称简称互"
    "换均算一致；(5) overall：四个维度任何一维不一致，整体即为否。")


def load_event_data():
    """事件任务数据源：优先用仓库内归档（archive/），/tmp 仅作兼容回退。"""
    src = None
    for cand in (Path(__file__).parent / "archive" / "event_verify_gen.py",
                 Path("/tmp/event_verify_gen.py")):
        if cand.exists():
            src = cand.read_text()
            break
    if src is None:
        raise FileNotFoundError("找不到 event_verify_gen.py（archive/ 与 /tmp 均无）")
    ns = {}
    exec(compile(src.split("\nrows = []")[0], "event_data", "exec"), ns)  # noqa: S102
    title, body, claims, chunks = ns["TITLE"], ns["BODY"], ns["C"], ns["DOC_CHUNKS"]
    for c in claims:  # 修 v1 标注过严：EV_A 族时间去掉推断年份
        c["time"] = c["time"].replace("2026年（今年）早些时候宣布", "今年早些时候宣布")
    return title, body, claims, chunks


def main():
    TITLE, BODY, C, DOC_CHUNKS = load_event_data()
    article_en = TITLE + "\n\n" + BODY
    (ROOT / "entity_event_tasks_v2.json").write_text(json.dumps({
        "article": article_en,
        "entities": [{"key": k, "name": n, "group": g, "type": t}
                     for k, n, g, t in ENTITIES_V2],
        "claims": C, "entity_rules": ENTITY_RULES, "event_rules": EVENT_RULES},
        ensure_ascii=False, indent=1))

    results = {}

    # ---------- Jev 实体（29 nouls 一请求，规则入 state）----------
    state = {"文章": {"标题": TITLE, "正文": BODY},
             "判定规则": ENTITY_RULES,
             "query_entities": {k: {"名称": n, "类型": t}
                                for k, n, g, t in ENTITIES_V2}}
    questions = {f"in_article_{k}": {
        "type": "noul",
        "instructions": (
            f"按 `判定规则` 判断 `query_entities.{k}` 所指的实体是否在 `文章` 中出现。"
            f"不要只做字面字符串匹配。"),
        "criteria": {"true": "文章标题或正文提及了该实体（含全称、简称、别名等"
                            "指代同一实体的写法）。",
                     "false": "文章未提及该实体；或字面相近但指向另一对象；或为"
                              "编造的相似名称；或仅因主题相关/同系列而被联想。"}}
        for k, n, g, t in ENTITIES_V2}
    r = or_decisions({"model": "typesafe/jev-1.13-20260917",
                      "state": state, "questions": questions})
    results["entity_jev"] = [
        {"entity": n, "group": g,
         "score": r["answers"][f"in_article_{k}"]["noul"]}
        for k, n, g, t in ENTITIES_V2]
    results["_entity_jev_usage"] = r.get("usage")
    print("jev entity done", flush=True)

    # ---------- Jev 事件（200 问一请求，规则入 state）----------
    state = {"文章": {"标题": TITLE, "正文": BODY}, "判定规则": EVENT_RULES,
             "claims": {c["id"]: {"发出者": c["issuer"], "承受者": c["receiver"],
                                  "关系": c["relation"], "时间": c["time"]}
                        for c in C}}
    questions = {}
    for c in C:
        i = c["id"]
        questions[f"{i}_overall"] = {
            "type": "noul",
            "instructions": (f"按 `判定规则` 整体判断 `claims.{i}` 与 `文章` 的记载"
                             f"是否一致：发出者、承受者、关系、时间四个维度全部相符"
                             f"才算一致，任何一维不符即为否。")}
        for d, zh in (("issuer", "发出者"), ("receiver", "承受者")):
            questions[f"{i}_{d}"] = {
                "type": "noul",
                "instructions": (f"仅判断{zh}维度：`claims.{i}.{zh}` 与 `文章` 记载"
                                 f"的该事件{zh}是否指同一实体。")}
        questions[f"{i}_relation"] = {
            "type": "noul",
            "instructions": (f"仅判断关系维度：`claims.{i}.关系` 所述的动作、金额与"
                             f"性质是否与 `文章` 记载一致（按判定规则：金额不符即否，"
                             f"上限与固定值不等价）。")}
        questions[f"{i}_time"] = {
            "type": "noul",
            "instructions": (f"仅判断时间维度：`claims.{i}.时间` 是否与 `文章` 的记"
                             f"载一致（按判定规则：文章未给出时间而声明给出具体时间"
                             f"判否；相对时间表述一致判是）。")}
    r = or_decisions({"model": "typesafe/jev-1.13-20260917",
                      "state": state, "questions": questions})
    ev = {q: a["noul"] for q, a in r["answers"].items()}
    results["event_jev"] = [
        {"id": c["id"], "type": c["type"],
         **{d: ev.get(f"{c['id']}_{d}") for d in
            ("overall", "issuer", "receiver", "relation", "time")}}
        for c in C]
    results["_event_jev_usage"] = r.get("usage")
    print("jev event done", flush=True)

    # ---------- 8B 实体 ----------

    def ent_8b(row):
        _, name, _, _ = row
        prompt = (f"文章：\n{article_en[:3500]}\n\n{ENTITY_RULES}\n\n"
                  f"判断实体「{name}」是否在这篇文章中出现。只输出一个 0 到 100 的"
                  f"整数，表示该实体在文中出现的把握（100=肯定出现，0=肯定没出现）。")
        out = chat(prompt, max_tokens=16)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        m = re.search(r"\d+", out)
        return int(m.group()) / 100 if m else None

    with ThreadPoolExecutor(max_workers=3) as ex:
        scores = list(ex.map(ent_8b, ENTITIES_V2))
    results["entity_8b"] = [{"entity": r[1], "group": r[2], "score": s}
                            for r, s in zip(ENTITIES_V2, scores)]
    print("8b entity done", flush=True)

    # ---------- 8B 事件 ----------
    def ev_8b(c):
        claim = (f"发出者：{c['issuer']}；承受者：{c['receiver']}；"
                 f"关系：{c['relation']}；时间：{c['time']}")
        prompt = (f"文章：\n{article_en[:3000]}\n\n{EVENT_RULES}\n\n"
                  f"待核对事件声明：{claim}\n\n"
                  f"分别判断该声明 overall/issuer/receiver/relation/time 五个方面与"
                  f"文章记载是否一致（按上述规则）。只输出 JSON："
                  f"{{\"overall\":x,\"issuer\":x,\"receiver\":x,\"relation\":x,"
                  f"\"time\":x}}，x 为 0 到 100 的整数把握分。")
        out = chat(prompt, max_tokens=64)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        m = re.search(r"\{.*\}", out, flags=re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group())
            return {k: d.get(k, 50) / 100 for k in
                    ("overall", "issuer", "receiver", "relation", "time")}
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        ev_scores = list(ex.map(ev_8b, C))
    results["event_8b"] = [{"id": c["id"], "type": c["type"], **(s or {})}
                           for c, s in zip(C, ev_scores)]
    print("8b event done", flush=True)

    # ---------- Embedding 实体/事件 ----------
    ent_vecs, _ = embed([r[1] for r in ENTITIES_V2] + [article_en[:3000]])
    av = ent_vecs[-1]
    results["entity_embed"] = [{"entity": r[1], "group": r[2],
                                "score": round(cosine(v, av), 4)}
                               for r, v in zip(ENTITIES_V2, ent_vecs[:-1])]
    ev_vecs, _ = embed([c["en"] for c in C] + [article_en[:3000]])
    av2 = ev_vecs[-1]
    results["event_embed"] = [{"id": c["id"], "type": c["type"],
                               "score": round(cosine(v, av2), 4)}
                              for c, v in zip(C, ev_vecs[:-1])]
    print("embed done", flush=True)

    # ---------- Rerank 实体/事件 ----------
    def rr_ent(row):
        _, name, _, _ = row
        return max(rerank(name, DOC_CHUNKS))

    with ThreadPoolExecutor(max_workers=2) as ex:
        rs = list(ex.map(rr_ent, ENTITIES_V2))
    results["entity_rerank"] = [{"entity": r[1], "group": r[2],
                                 "score": round(s, 4)}
                                for r, s in zip(ENTITIES_V2, rs)]

    def rr_ev(c):
        zh = f"{c['issuer']}{c['relation']}，涉及{c['receiver']}；时间：{c['time']}。"
        return round(max(rerank(zh, DOC_CHUNKS)), 4)

    with ThreadPoolExecutor(max_workers=2) as ex:
        rs = list(ex.map(rr_ev, C))
    results["event_rerank"] = [{"id": c["id"], "type": c["type"], "score": s}
                               for c, s in zip(C, rs)]
    print("rerank done", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "extra_entity_event.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
