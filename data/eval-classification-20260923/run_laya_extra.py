#!/usr/bin/env python3
"""Laya 实体/事件 v2：客户端显式截断，每请求 ≤1024 token。

实体：29 个分 5 批（每批 6 个 noul），文章截前 600 字。
事件：每声明一个请求（5 个 noul），规则 + 声明 + 文章截前 500 字。
判定规则与 Jev 版相同（entity_rules/event_rules），为省预算省略每问 criteria。
"""
import json

from common import ROOT, http_json

LAYA_URL = "http://100.92.114.89:8084/v1/systemone"
ART_CAP_ENT = 600
ART_CAP_EV = 500


def main():
    task = json.loads((ROOT / "entity_event_tasks_v2.json").read_text())
    article, C, ents = task["article"], task["claims"], task["entities"]
    TITLE, BODY = article.split("\n\n", 1)
    ent_state_art = {"标题": TITLE, "正文": BODY[:ART_CAP_ENT]}
    ev_state_art = {"标题": TITLE, "正文": BODY[:ART_CAP_EV]}

    f = ROOT / "results" / "extra_entity_event.json"
    results = json.loads(f.read_text())
    # 备份暗截断版
    results["entity_laya_flat"] = results.get("entity_laya")
    results["event_laya_flat"] = results.get("event_laya")

    # ---- 实体：分批 ----
    ent_out = []
    B = 6
    for s in range(0, len(ents), B):
        batch = ents[s:s + B]
        state = {"文章": ent_state_art, "判定规则": task["entity_rules"],
                 "query_entities": {e["key"]: {"名称": e["name"], "类型": e["type"]}
                                    for e in batch}}
        questions = {f"in_article_{e['key']}": {
            "type": "noul",
            "instructions": (f"按 `判定规则` 判断 `query_entities.{e['key']}` "
                             f"是否在 `文章` 中出现。")} for e in batch}
        r = http_json(LAYA_URL, {"model": "multilingual", "state": state,
                                 "questions": questions}, timeout=300)
        for e in batch:
            ent_out.append({"entity": e["name"], "group": e["group"],
                            "score": r["answers"][f"in_article_{e['key']}"]["noul"]})
        print(f"ent [{s+B}/{len(ents)}] tok={r['usage'].get('input_tokens')}",
              flush=True)
    results["entity_laya"] = ent_out
    print("laya entity done", flush=True)

    # ---- 事件：每声明一请求 ----
    ev_out = []
    for c in C:
        state = {"文章": ev_state_art, "判定规则": task["event_rules"],
                 "声明": {"发出者": c["issuer"], "承受者": c["receiver"],
                          "关系": c["relation"], "时间": c["time"]}}
        questions = {}
        for d, zh in (("overall", "整体（四维全部相符才算一致）"),
                      ("issuer", "发出者维度"), ("receiver", "承受者维度"),
                      ("relation", "关系维度"), ("time", "时间维度")):
            questions[d] = {
                "type": "noul",
                "instructions": (f"按 `判定规则` 判断 `声明` 与 `文章` 的{zh}"
                                 f"是否一致。")}
        r = http_json(LAYA_URL, {"model": "multilingual", "state": state,
                                 "questions": questions}, timeout=300)
        a = r["answers"]
        ev_out.append({"id": c["id"], "type": c["type"],
                       **{d: a[d]["noul"] for d in
                          ("overall", "issuer", "receiver", "relation", "time")}})
        print(f"ev {c['id']} tok={r['usage'].get('input_tokens')}", flush=True)
    results["event_laya"] = ev_out
    print("laya event done", flush=True)

    f.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
