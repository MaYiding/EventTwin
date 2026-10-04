#!/usr/bin/env python3
"""事件核对大规模测试：生成 Jev 请求体 + 跑 rerank 对比。

40 条事件声明（5 个基准事件 × 各维度分程度伪造），每条 5 个 noul 问题
（overall/issuer/receiver/relation/time），共 200 问。
rerank 侧：同一声明以中/英两种 query 对文章 3 个分块取最大分。
"""

import json
import time
import urllib.request

BASE_URL = "http://100.92.114.89:8083/v1/rerank"
MODEL = "qwen3-reranker-0.6b"

TITLE = "Apple settlement offers eligible iPhone owners up to $95. Here is how to file a claim."
BODY = """The claim window is now open for an Apple settlement announced earlier this year, giving certain iPhone users the chance to get up to $95.

Apple agreed to pay $250 million to settle claims that it misled customers about Siri's Apple Intelligence features. Customers who were part of the class action lawsuit said they purchased iPhones expecting certain Siri features they allegedly did not receive.

"The $250 million settlement fund represents the largest false advertising settlement in history," Ryan Clarkson, co-lead counsel on the class-action lawsuit and founder and managing partner of Clarkson Law Firm, said in an email to CBS News.

Apple declined to comment. The company has previously denied the allegations. According to the settlement agreement, the settlement shall "not be construed in any fashion as an admission of liability or wrongdoing by Apple."

Here's what to know about eligibility and how to file a claim.

How do I file a claim?

iPhone users can visit SmartphoneAISettlement.com to file a claim, according to a spokesperson for Clarkson Law Firm, a public interest law firm.

The person must input basic personal information, including their name, address and phone number. They must also enter their iPhone's serial number, which can be found in the phone's Settings section.

Once they finish inputting that information, they can select the payment method. The options are PayPal, Venmo, direct deposit or check.

As SmartphoneAISettlement.com points out, consumers are limited to "one cash payment per eligible device." If they purchased another iPhone that meets the eligibility requirements, they must complete a separate claim form.

How do I know if I am eligible?

If you bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 10, 2024, and March 29, 2025, you're eligible to file a claim.

According to SmartphoneAISettlement.com, the person filing a claim must be the person who originally purchased the device. The phone must also be for personal or professional use, not for resale.

If you're not sure if you're eligible, you can call 1-888-988-8945 or write to this address for information: Landsheft, et al. v. Apple Inc.; Settlement Administrator; P.O. Box 301132; Los Angeles, CA 90030-1132.

How long do I have to file a claim?

The claims window is open until Dec. 21, 2026.

How much money will I receive?

Customers who submit a valid claim form will initially get $25 per eligible device. However, that amount may be increased up to $95 or decreased depending on the "total number of valid claims submitted and other factors," the claims website says.

When will I get the payment?

A district court in San Jose, California, will hold a hearing on Feb. 24, 2027, to decide whether to approve the settlement. If there are any appeals, it could delay the payment process.

"If there is no appeal, your settlement benefit will be processed promptly," according to the claim website."""

DOC_FULL = TITLE + "\n\n" + BODY
CHUNK_SIZE, CHUNK_OVERLAP = 1100, 50
_chunks, _s = [], 0
while _s < len(DOC_FULL):
    _chunks.append(DOC_FULL[_s:_s + CHUNK_SIZE])
    _s += CHUNK_SIZE - CHUNK_OVERLAP
DOC_CHUNKS = _chunks

# ---------------------------------------------------------------------------
# 40 条事件声明。type: TRUE=与文章一致 / 伪造维度; note 中文备注
# ---------------------------------------------------------------------------
C = []

def add(id_, typ, note, issuer, receiver, relation, time_, en):
    C.append(dict(id=id_, type=typ, note=note, issuer=issuer,
                  receiver=receiver, relation=relation, time=time_, en=en))

# ---- EV_A 和解基金：S=Apple, R=集体诉讼原告, L=支付$250M和解, T=今年早些时候宣布 ----
add("c01", "TRUE", "基准真值",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付2.5亿美元设立和解基金，了结关于其就 Siri 与 Apple Intelligence 功能误导消费者的指控",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay $250 million to settle claims that it misled customers about Siri's Apple Intelligence features; the settlement was announced earlier this year.")
add("c02", "issuer|同类替换", "Google 顶替 Apple",
    "Google", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付2.5亿美元设立和解基金，了结关于其就 Siri 与 Apple Intelligence 功能误导消费者的指控",
    "2026年（今年）早些时候宣布",
    "Google agreed to pay $250 million to settle claims that it misled customers about Siri's Apple Intelligence features; the settlement was announced earlier this year.")
add("c03", "issuer|近似拼写", "Appel 伪造",
    "Appel", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付2.5亿美元设立和解基金，了结关于其就 Siri 与 Apple Intelligence 功能误导消费者的指控",
    "2026年（今年）早些时候宣布",
    "Appel agreed to pay $250 million to settle claims that it misled customers about Siri's Apple Intelligence features; the settlement was announced earlier this year.")
add("c04", "issuer+receiver|角色对调", "原被告互换",
    "集体诉讼原告", "Apple",
    "同意支付2.5亿美元和解金了结诉讼",
    "2026年（今年）早些时候宣布",
    "The class-action plaintiffs agreed to pay Apple $250 million to settle the lawsuit; announced earlier this year.")
add("c05", "receiver|同类替换", "Android 用户",
    "Apple", "Android 手机用户",
    "同意支付2.5亿美元设立和解基金，了结关于其就 Siri 与 Apple Intelligence 功能误导消费者的指控",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay $250 million to settle claims that it misled Android phone users about Siri's Apple Intelligence features; announced earlier this year.")
add("c06", "receiver|字面歧义", "Apple=苹果水果 陷阱",
    "Apple", "苹果种植农户协会",
    "同意支付2.5亿美元和解金",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay $250 million to settle claims brought by an association of apple orchard farmers; announced earlier this year.")
add("c07", "relation|金额近似(±4%)", "2.5亿→2.4亿",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付2.4亿美元和解金",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay $240 million to settle the claims; announced earlier this year.")
add("c08", "relation|数量级(×10)", "2.5亿→25亿",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付25亿美元和解金",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay $2.5 billion to settle the claims; announced earlier this year.")
add("c09", "relation|总额/单价混淆", "总额95美元",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付总额最高95美元的和解金",
    "2026年（今年）早些时候宣布",
    "Apple agreed to pay a total of up to $95 to settle the claims; announced earlier this year.")
add("c10", "relation|极性反转", "拒付+否认",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "拒绝支付和解金并否认所有指控",
    "2026年（今年）早些时候宣布",
    "Apple refused to pay any settlement and denied all the allegations.")
add("c11", "time|具体化矛盾", "文章只说今年早些时候",
    "Apple", "集体诉讼原告（符合条件的 iPhone 用户）",
    "同意支付2.5亿美元设立和解基金，了结关于其就 Siri 与 Apple Intelligence 功能误导消费者的指控",
    "2025年11月宣布",
    "Apple agreed to pay $250 million to settle the claims; the settlement was announced in November 2025.")

# ---- EV_B 听证会：S=圣何塞联邦地区法院, R=和解协议, L=听证决定批准, T=2027-02-24 ----
add("c12", "TRUE", "基准真值",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2027年2月24日",
    "A district court in San Jose, California, will hold a hearing on Feb. 24, 2027, to decide whether to approve the settlement.")
add("c13", "TRUE|时间格式等价", "同一日期英文写法",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "February 24, 2027",
    "A district court in San Jose, California, will hold a hearing on February 24, 2027, to decide whether to approve the settlement.")
add("c14", "time|±1天", "2/24→2/25",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2027年2月25日",
    "A district court in San Jose, California, will hold a hearing on Feb. 25, 2027, to decide whether to approve the settlement.")
add("c15", "time|±1月", "2/24→3/24",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2027年3月24日",
    "A district court in San Jose, California, will hold a hearing on Mar. 24, 2027, to decide whether to approve the settlement.")
add("c16", "time|±1年", "2027→2026",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2026年2月24日",
    "A district court in San Jose, California, will hold a hearing on Feb. 24, 2026, to decide whether to approve the settlement.")
add("c17", "time|跨事件混淆", "把索赔截止日安到听证会",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2026年12月21日",
    "A district court in San Jose, California, will hold a hearing on Dec. 21, 2026, to decide whether to approve the settlement.")
add("c18", "issuer|同类城市", "圣何塞→洛杉矶(文中出现)",
    "加利福尼亚州洛杉矶的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2027年2月24日",
    "A district court in Los Angeles, California, will hold a hearing on Feb. 24, 2027, to decide whether to approve the settlement.")
add("c19", "issuer|性质近似", "联邦→州法院",
    "加利福尼亚州圣何塞的州法院", "Apple 和解协议",
    "举行听证会以决定是否批准和解",
    "2027年2月24日",
    "A state court in San Jose, California, will hold a hearing on Feb. 24, 2027, to decide whether to approve the settlement.")
add("c20", "relation|极性反转", "批准→驳回",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "举行听证会以决定是否驳回和解",
    "2027年2月24日",
    "A district court in San Jose, California, will hold a hearing on Feb. 24, 2027, to decide whether to reject the settlement.")
add("c21", "TRUE|关系等价改写", "语义等价",
    "加利福尼亚州圣何塞的联邦地区法院", "Apple 和解协议",
    "开庭审理以裁定是否批准该和解",
    "2027年2月24日",
    "A district court in San Jose, California, will open a hearing to rule on whether to approve the settlement on Feb. 24, 2027.")

# ---- EV_C 索赔：S=符合条件用户, R=和解管理方, L=申请最高$95/台, T=截止2026-12-21 ----
add("c22", "TRUE", "基准真值",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2026年12月21日",
    "Eligible iPhone users can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Dec. 21, 2026.")
add("c23", "time|±1天", "12/21→12/22",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2026年12月22日",
    "Eligible iPhone users can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Dec. 22, 2026.")
add("c24", "time|跨事件混淆", "听证会日期",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2027年2月24日",
    "Eligible iPhone users can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Feb. 24, 2027.")
add("c25", "time|±1年", "2026→2027",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2027年12月21日",
    "Eligible iPhone users can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Dec. 21, 2027.")
add("c26", "relation|上限→固定", "最高95→固定95",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "每台设备固定获得95美元",
    "索赔窗口截止2026年12月21日",
    "Eligible users will receive a fixed payment of $95 per device; the claims window closes Dec. 21, 2026.")
add("c27", "relation|初始额混淆", "25 当最终额",
    "符合条件的 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "每台设备固定获得25美元",
    "索赔窗口截止2026年12月21日",
    "Eligible users will receive a fixed payment of $25 per device; the claims window closes Dec. 21, 2026.")
add("c28", "issuer|范围放宽", "限定→所有用户",
    "所有 iPhone 用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2026年12月21日",
    "All iPhone users can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Dec. 21, 2026.")
add("c29", "issuer|型号近似", "14 Pro 不在范围",
    "购买了 iPhone 14 Pro 的用户", "和解管理方（SmartphoneAISettlement.com）",
    "提交索赔申请，每台设备最高可获95美元",
    "索赔窗口截止2026年12月21日",
    "Users who bought an iPhone 14 Pro can submit a claim to the settlement administrator for up to $95 per device; the claims window closes Dec. 21, 2026.")

# ---- EV_D 诉讼：S=Landsheft等, R=Apple Inc., L=集体诉讼, T=未提及 ----
add("c30", "TRUE", "基准真值(时间缺失)",
    "Landsheft 等", "Apple Inc.",
    "提起集体诉讼",
    "未提及",
    "Landsheft et al. filed a class-action lawsuit against Apple Inc.")
add("c31", "TRUE|承受者等价", "Apple 无 Inc.",
    "Landsheft 等", "Apple",
    "提起集体诉讼",
    "未提及",
    "Landsheft et al. filed a class-action lawsuit against Apple.")
add("c32", "issuer|角色混淆", "律师当原告",
    "Ryan Clarkson 等", "Apple Inc.",
    "提起集体诉讼",
    "未提及",
    "Ryan Clarkson et al. filed a class-action lawsuit against Apple Inc.")
add("c33", "time|无中生有", "文章未提时间",
    "Landsheft 等", "Apple Inc.",
    "提起集体诉讼",
    "2025年6月提起",
    "Landsheft et al. filed a class-action lawsuit against Apple Inc. in June 2025.")
add("c34", "relation|性质替换", "民事→刑事",
    "Landsheft 等", "Apple Inc.",
    "对 Apple 提起刑事诉讼",
    "未提及",
    "Landsheft et al. brought criminal charges against Apple Inc.")

# ---- EV_E 购买窗口：S=消费者, R=特定机型, L=购买→有资格, T=2024-06-10~2025-03-29 ----
add("c35", "TRUE", "基准真值",
    "消费者", "iPhone 15 Pro、iPhone 15 Pro Max 或任意 iPhone 16 机型",
    "在此期间购买即具备索赔资格",
    "2024年6月10日至2025年3月29日",
    "Customers who bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 10, 2024 and March 29, 2025 are eligible to file a claim.")
add("c36", "time|起点+1天", "6/10→6/11",
    "消费者", "iPhone 15 Pro、iPhone 15 Pro Max 或任意 iPhone 16 机型",
    "在此期间购买即具备索赔资格",
    "2024年6月11日至2025年3月29日",
    "Customers who bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 11, 2024 and March 29, 2025 are eligible to file a claim.")
add("c37", "time|终点-1天", "3/29→3/28",
    "消费者", "iPhone 15 Pro、iPhone 15 Pro Max 或任意 iPhone 16 机型",
    "在此期间购买即具备索赔资格",
    "2024年6月10日至2025年3月28日",
    "Customers who bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 10, 2024 and March 28, 2025 are eligible to file a claim.")
add("c38", "time|双年份前移", "整体前移一年",
    "消费者", "iPhone 15 Pro、iPhone 15 Pro Max 或任意 iPhone 16 机型",
    "在此期间购买即具备索赔资格",
    "2023年6月10日至2024年3月29日",
    "Customers who bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model between June 10, 2023 and March 29, 2024 are eligible to file a claim.")
add("c39", "receiver|型号近似", "14 Pro/15 不在范围",
    "消费者", "iPhone 14 Pro 或任意 iPhone 15 机型",
    "在此期间购买即具备索赔资格",
    "2024年6月10日至2025年3月29日",
    "Customers who bought an iPhone 14 Pro or any iPhone 15 model between June 10, 2024 and March 29, 2025 are eligible to file a claim.")
add("c40", "relation|添加限定", "官方渠道(文章未限定)",
    "消费者", "iPhone 15 Pro、iPhone 15 Pro Max 或任意 iPhone 16 机型",
    "从 Apple 官方渠道购买即具备索赔资格",
    "2024年6月10日至2025年3月29日",
    "Customers who bought an iPhone 15 Pro, iPhone 15 Pro Max, or any iPhone 16 model from Apple's official channels between June 10, 2024 and March 29, 2025 are eligible to file a claim.")

assert len(C) == 40, len(C)

# ---------------------------------------------------------------------------
# 生成 Jev 请求体
# ---------------------------------------------------------------------------
def claim_state(c):
    return {"发出者": c["issuer"], "承受者": c["receiver"],
            "关系": c["relation"], "时间": c["time"]}

questions = {}
for c in C:
    i = c["id"]
    questions[f"{i}_overall"] = {
        "type": "noul",
        "instructions": f"整体判断事件记录 `claims.{i}` 与 `article` 的记载是否一致：发出者、承受者、关系、时间四个维度全部相符才算一致。",
        "criteria": {
            "true": "四个维度均与文章一致；全称/简称/别名称等价、同一日期不同书写格式均算一致。",
            "false": "任一维度与文章矛盾，或添加了文章不支持的限定条件，或对文章未提及的信息虚构了内容。"
        },
    }
    questions[f"{i}_issuer"] = {
        "type": "noul",
        "instructions": f"仅判断发出者维度：`claims.{i}.发出者` 与 `article` 记载的该事件发出者是否指同一实体。",
    }
    questions[f"{i}_receiver"] = {
        "type": "noul",
        "instructions": f"仅判断承受者维度：`claims.{i}.承受者` 与 `article` 记载的该事件承受者是否指同一实体。",
    }
    questions[f"{i}_relation"] = {
        "type": "noul",
        "instructions": f"仅判断关系维度：`claims.{i}.关系` 所述的动作、金额与性质是否与 `article` 记载一致。",
    }
    questions[f"{i}_time"] = {
        "type": "noul",
        "instructions": f"仅判断时间维度：`claims.{i}.时间` 是否与 `article` 的记载一致。",
        "criteria": {
            "true": "与文章记载一致；文章未给出时间而此处也未给出具体时间，同样算一致。",
            "false": "与文章记载的时间矛盾；或文章未给出时间而此处给出了具体时间。"
        },
    }

jev_body = {
    "model": "jev-latest",
    "state": {
        "article": {
            "标题": TITLE,
            "正文": "【原样粘贴正文，与之前 22 实体测试同一篇文章，一字不改】",
        },
        "claims": {c["id"]: claim_state(c) for c in C},
    },
    "questions": questions,
}
with open("/tmp/jev_event_body.json", "w") as f:
    json.dump(jev_body, f, ensure_ascii=False, indent=1)
print(f"Jev body: /tmp/jev_event_body.json  "
      f"({len(questions)} questions, {len(C)} claims, "
      f"{len(json.dumps(jev_body, ensure_ascii=False))} chars)")

# ---------------------------------------------------------------------------
# rerank 测试：中/英 query × 3 块取 max
# ---------------------------------------------------------------------------
def rerank(query, documents):
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps({"model": MODEL, "query": query,
                         "documents": documents, "top_n": len(documents)}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read())
    scores = [0.0] * len(documents)
    for r in body.get("results", []):
        scores[r.get("index", 0)] = float(r.get("relevance_score", 0.0))
    return scores

def zh_query(c):
    return (f"{c['issuer']}{c['relation']}，涉及{c['receiver']}；"
            f"时间：{c['time']}。")

rows = []
for c in C:
    zh_max = max(rerank(zh_query(c), DOC_CHUNKS))
    time.sleep(0.1)
    en_max = max(rerank(c["en"], DOC_CHUNKS))
    time.sleep(0.1)
    rows.append({"id": c["id"], "type": c["type"], "note": c["note"],
                 "zh_max": round(zh_max, 4), "en_max": round(en_max, 4)})
    print(f"{c['id']} [{c['type']:<22}] zh={zh_max:.4f}  en={en_max:.4f}", flush=True)

with open("/tmp/event_rerank_results.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

# 分组统计
print("\n=== 分组均值（zh/en 最大块得分）===")
groups = {"TRUE": [], "time|": [], "issuer|": [], "receiver|": [], "relation|": []}
for r in rows:
    for g in groups:
        if r["type"] == "TRUE" and g == "TRUE":
            groups[g].append(r)
        elif g != "TRUE" and r["type"].startswith(g):
            groups[g].append(r)
for g, rs in groups.items():
    if not rs:
        continue
    zm = sum(x["zh_max"] for x in rs) / len(rs)
    em = sum(x["en_max"] for x in rs) / len(rs)
    print(f"{g:<12} n={len(rs):<2} zh_mean={zm:.4f}  en_mean={em:.4f}")

print("\n=== 关键对照组（zh_max）===")
pairs = [("c12 TRUE 基准", "c12"), ("c14 ±1天", "c14"), ("c15 ±1月", "c15"),
         ("c16 ±1年", "c16"), ("c35 TRUE 基准", "c35"), ("c36 起点+1天", "c36"),
         ("c37 终点-1天", "c37"), ("c01 TRUE 基准", "c01"), ("c07 金额±4%", "c07"),
         ("c08 金额×10", "c08"), ("c09 总额混淆", "c09")]
idx = {r["id"]: r for r in rows}
for label, cid in pairs:
    r = idx[cid]
    print(f"{label:<16} zh={r['zh_max']:.4f}  en={r['en_max']:.4f}")
