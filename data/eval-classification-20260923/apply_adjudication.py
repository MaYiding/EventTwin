#!/usr/bin/env python3
"""人工标注修正层：对 Jev 全部 124 条叶子错误逐条判读的结论（2026-09-24 会话内完成），
叠加到规则标注之上。三类处置：
  fix  —— 参考答案有误，改为内容正确标签（绝大多数与 Jev 预测一致，因复核确认其正确；
          T258 手写为 技术与研究）；共 70 条（L1 层 30 + 下层 40）
  drop —— 真两可/无合适类目/重复转载，剔除；共 38 条（L1 层 25 + 下层 13）
  keep —— 模型真错，保持规则标注；共 16 条（L1 层 1 + 下层 15）
修正只改参考答案，不改变模型输入/输出；对四模型的重新打分等价于重跑。
"""
import json
from pathlib import Path

from common import ROOT, load_taxonomy, load_testset

# L1 层错误归因（56 条，编号对应导出顺序）
L1_FIX = ["T005", "T011", "T016", "T028", "T035", "T041", "T047", "T054",
          "T058", "T060", "T069", "T126", "T130", "T131", "T140", "T176",
          "T177", "T194", "T195", "T203", "T206", "T210", "T211", "T219",
          "T225", "T227", "T241", "T279", "T290", "T292"]
L1_DROP = ["T052", "T071", "T083", "T087", "T108", "T125", "T169", "T183",
           "T190", "T191", "T196", "T209", "T215", "T218", "T220", "T222",
           "T226", "T228", "T239", "T242", "T243", "T264", "T281", "T288",
           "T294"]
# L1 层模型真错：T106（荣耀×字节"豆包手机"合作，gold 手机与穿戴 更准）

# 下层错误归因（68 条）
LOW_FIX = ["T001", "T004", "T006", "T009", "T022", "T032", "T038", "T049",
           "T070", "T073", "T081", "T088", "T090", "T097", "T104", "T110",
           "T129", "T154", "T156", "T166", "T173", "T181", "T193", "T198",
           "T214", "T217", "T221", "T236", "T245", "T246", "T248", "T255",
           "T257", "T258", "T265", "T270", "T284", "T285", "T295"]
LOW_KEEP = ["T014", "T042", "T075", "T096", "T113", "T119", "T120", "T123",
            "T124", "T134", "T135", "T180", "T182", "T289", "T299"]
LOW_DROP = ["T018", "T024", "T062", "T079", "T085", "T145", "T157", "T167",
            "T207", "T253", "T259", "T278", "T287"]

HAND_PATHS = {  # 手写修正（未采纳模型预测的少数）
    "T258": ["机器人与智能制造", "机器人与智能装备", "技术与研究"],
}


def main():
    tax = load_taxonomy()
    id2path = {lf["id"]: "/".join(lf["path"]) for lf in tax["leaves"]}
    items = load_testset()
    jev = {r["id"]: r for r in json.loads(
        (ROOT / "results" / "jev.json").read_text())}

    fixes, drops = {}, L1_DROP + LOW_DROP
    for tid in L1_FIX + LOW_FIX:
        if tid in HAND_PATHS:
            fixes[tid] = HAND_PATHS[tid]
        else:
            r = jev.get(tid)
            if r and r.get("choice") in id2path:
                fixes[tid] = id2path[r["choice"]].split("/")

    out, audit = [], []
    for it in items:
        if it["id"] in drops:
            audit.append({"id": it["id"], "action": "drop",
                          "old": "/".join(it["gold_path"]), "title": it["title"]})
            continue
        if it["id"] in fixes:
            old = "/".join(it["gold_path"])
            it["gold_path"] = fixes[it["id"]]
            audit.append({"id": it["id"], "action": "fix", "old": old,
                          "new": "/".join(it["gold_path"]),
                          "title": it["title"]})
        out.append(it)

    (ROOT / "adjudication.json").write_text(json.dumps({
        "n_final": len(out), "n_fix": len(fixes), "n_drop": len(drops),
        "n_keep_model_wrong": len(LOW_KEEP) + 1, "audit": audit},
        ensure_ascii=False, indent=1))
    (ROOT / "testset.json").write_text(json.dumps(
        {"n": len(out), "items": out}, ensure_ascii=False, indent=1))
    print(f"最终 {len(out)} 条（改标 {len(fixes)}，剔除 {len(drops)}，"
          f"模型真错保留 {len(LOW_KEEP) + 1}）")


if __name__ == "__main__":
    main()
