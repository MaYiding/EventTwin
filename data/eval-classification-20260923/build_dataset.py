#!/usr/bin/env python3
"""v2 数据构建：内容优先标注 + 更通用体系 + 明确判定规则。

与 v1 的区别：
1. ground truth 按文章内容归档（不按公司主业务）：领域关键词规则优先，
   无命中时回退公司主标签（此时内容与主业一致）。
2. 体系扩充：新增 L1 机器人与智能制造、其他与跨界；新增 L2 金融与支付；
   新增 L3 财报与业绩、产能与扩张。共 7 L1 / 18 L2 / 10 L3 / 252 叶子。
3. 修掉 v1 关键词陷阱：发布会带售价→产品发布；"发布财报"→财报与业绩；
   "投资某地建厂"→产能与扩张；"鸿蒙智行/应用于"→硬件；高管落马→法律与监管。
4. 丢弃真两可样本：早参/日报等多主题简报、双信号无法判主次的标题。
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(20260923)
ROOT = Path(__file__).parent
CORPUS = ROOT.parent / "corpus" / "clean"

# ---------------------------------------------------------------- 体系
L1_LIST = ["消费电子", "新能源汽车", "半导体", "互联网平台", "软件与AI",
           "机器人与智能制造", "其他与跨界"]
L2_TREE = {
    "消费电子": ["手机与穿戴", "电脑与平板", "影像与无人机", "智能家居"],
    "新能源汽车": ["整车制造", "动力电池", "自动驾驶技术"],
    "半导体": ["芯片制造", "芯片设计"],
    "互联网平台": ["电商零售", "社交内容", "本地生活", "游戏", "金融与支付"],
    "软件与AI": ["大模型与AI应用", "智能语音", "云与企业IT"],
    "机器人与智能制造": ["机器人与智能装备"],
    "其他与跨界": ["跨界与其他业务"],
}
L2_DESC = {
    "手机与穿戴": "手机/手表/耳机/眼镜等个人智能终端",
    "电脑与平板": "笔记本/PC/平板",
    "影像与无人机": "相机/无人机/影像设备",
    "智能家居": "家电/智能家居/个护小家电",
    "整车制造": "整车企业的新车、交付销量、经营动态",
    "动力电池": "电池/电芯/储能电池",
    "自动驾驶技术": "智驾/辅助驾驶/无人驾驶技术",
    "芯片制造": "晶圆制造/产能/工艺/面板制造",
    "芯片设计": "芯片设计与研发进展",
    "电商零售": "电商平台/网购/大促",
    "社交内容": "社区/短视频/直播/音乐/影视内容平台",
    "本地生活": "外卖/到店/即时零售",
    "游戏": "游戏/手游/电竞",
    "大模型与AI应用": "大模型/AI应用/生成式AI",
    "智能语音": "语音技术与产品",
    "云与企业IT": "云计算/服务器/数据中心",
    "金融与支付": "支付牌照/金融科技业务",
    "机器人与智能装备": "人形机器人/机械臂/智能装备",
    "跨界与其他业务": "无法归入上述领域的跨界或新兴业务",
}
L3_LIST = ["产品发布", "价格与销量", "专利与知识产权", "资本与并购",
           "合作与联盟", "人事与组织", "法律与监管", "技术与研究",
           "财报与业绩", "产能与扩张"]
L3_DESC = {
    "产品发布": "新品发布、上市、开售（发布会公布的售价信息仍属产品发布）",
    "价格与销量": "降价/涨价/调价；销量、交付量、出货量、份额数据",
    "专利与知识产权": "专利、商标、知识产权",
    "资本与并购": "收购并购、投资融资、入股、增减持（不含建厂投资）",
    "合作与联盟": "战略合作、签约、联手、联盟",
    "人事与组织": "任命、辞职、加盟、裁员、组织调整（不含贪腐被查）",
    "法律与监管": "诉讼、侵权、监管调查、反垄断、处罚（含高管落马/职务犯罪）",
    "技术与研究": "研发突破、新工艺、论文、技术进展",
    "财报与业绩": "季报年报、营收、利润、业绩",
    "产能与扩张": "建厂、投产、产能扩张、新基地落地",
}
L4_TREE = {
    "产品发布": {"硬件": "实体产品：手机/车型/电池/机器人等",
               "软件与服务": "系统/大模型/App/云服务（品牌名含鸿蒙、智行不改变硬件属性）"},
    "法律与监管": {"诉讼纠纷": "诉讼、侵权、禁令、判决",
                "监管调查": "调查、反垄断、处罚、职务犯罪"},
    "资本与并购": {"收购并购": "取得控制权的收购、合并",
                "投资融资": "投资入股/融资IPO/增持减持退出"},
    "价格与销量": {"价格调整": "降价、涨价、调价",
                "销量数据": "销量、交付量、份额"},
}

COMPANY_L2 = {  # 回退用：内容无领域信号时的公司主标签
    "apple": ("消费电子", "手机与穿戴"), "huawei": ("消费电子", "手机与穿戴"),
    "xiaomi": ("消费电子", "手机与穿戴"), "honor": ("消费电子", "手机与穿戴"),
    "oppo": ("消费电子", "手机与穿戴"), "vivo": ("消费电子", "手机与穿戴"),
    "lenovo": ("消费电子", "电脑与平板"), "dji": ("消费电子", "影像与无人机"),
    "tesla": ("新能源汽车", "整车制造"), "byd": ("新能源汽车", "整车制造"),
    "nio": ("新能源汽车", "整车制造"), "lixiang": ("新能源汽车", "整车制造"),
    "xpeng": ("新能源汽车", "整车制造"), "geely": ("新能源汽车", "整车制造"),
    "chery": ("新能源汽车", "整车制造"), "gwm": ("新能源汽车", "整车制造"),
    "saic": ("新能源汽车", "整车制造"), "gac": ("新能源汽车", "整车制造"),
    "catl": ("新能源汽车", "动力电池"), "smic": ("半导体", "芯片制造"),
    "alibaba": ("互联网平台", "电商零售"), "jd": ("互联网平台", "电商零售"),
    "tencent": ("互联网平台", "社交内容"), "bytedance": ("互联网平台", "社交内容"),
    "kuaishou": ("互联网平台", "社交内容"), "bilibili": ("互联网平台", "社交内容"),
    "meituan": ("互联网平台", "本地生活"), "netease": ("互联网平台", "游戏"),
    "baidu": ("软件与AI", "大模型与AI应用"), "iflytek": ("软件与AI", "智能语音"),
}

# ----------------------------------------- 内容领域规则
# 两档：TITLE 规则只匹配标题（AI/云/金融/平台类业务，正文顺带提及不该翻类）；
#       BOTH 规则标题优先、无命中再扫正文前 500 字（硬件/车/半导体等实体词）。
DOMAIN_TITLE = [
    (("机器人与智能制造", "机器人与智能装备"),
     r"人形机器人|Optimus|机械臂|工业机器人|智能装备|(?<!扫地)机器人"),
    (("互联网平台", "本地生活"),
     r"外卖|骑手|到店|本地生活|即时零售|即时配送|无人机配送|空投|美团|饿了么|"
     r"肯德基|麦当劳|星巴克"),
    (("互联网平台", "游戏"), r"游戏|手游|电竞|互娱|网游|端游|Steam"),
    (("互联网平台", "电商零售"),
     r"电商|网购|双1?1|直播间带货|商城|淘宝|天猫|货架"),
    (("互联网平台", "社交内容"),
     r"短视频|直播|社区|创作者|视频网站|会员|影视|音乐|内容平台|弹幕|番剧"),
    (("互联网平台", "金融与支付"), r"支付牌照|支付业务|金融牌照|数字人民币"),
    (("软件与AI", "大模型与AI应用"),
     r"大模型|生成式|文生视频|文生图|AIGC|智能体|AI应用|人工智能"),
    (("软件与AI", "智能语音"), r"语音识别|智能语音|讯飞星火|语音大模型"),
    (("软件与AI", "云与企业IT"),
     r"阿里云|腾讯云|华为云|百度智能云|火山引擎|云服务|服务器|数据中心|智算|"
     r"上云"),
]
DOMAIN_BOTH = [
    (("消费电子", "手机与穿戴"),
     r"手机|手表|手环|耳机|眼镜|穿戴|机型|Mate\s?\d|iPhone|Pixel"),
    (("消费电子", "电脑与平板"), r"笔记本|[Pp][Cc]\b|电脑|平板|MagicBook"),
    (("消费电子", "影像与无人机"), r"无人机|相机|影像器材|云台"),
    (("消费电子", "智能家居"), r"智能家居|家电|扫地机器人|电视|徕芬|吹风机"),
    (("半导体", "芯片设计"), r"芯片设计|自研芯片|麒麟\d|芯片研发|GPU芯片|AI芯片"),
    (("新能源汽车", "动力电池"),
     r"动力电池|刀片电池|电芯|储能电池|电池技术|固态电池|电池.{0,8}(量产|产能|工厂|投产|扩产)"),
    (("新能源汽车", "整车制造"),
     r"车型|汽车|轿车|SUV|MPV|交付量|车企|造车|新车|上市|增程|混动|"
     r"充电桩|充电站|超充|补能"),
    (("新能源汽车", "自动驾驶技术"),
     r"自动驾驶|无人驾驶|辅助驾驶|智驾技术|端到端大模型|CVPR|Robotaxi"),
    (("半导体", "芯片制造"),
     r"晶圆|光刻|制程|封装|面板|京东方|"
     r"(芯片|半导体|晶圆厂).{0,8}(量产|产能|工厂|投产|扩产)"),
    (("新能源汽车", "动力电池"),
     r"动力电池|刀片电池|电芯|储能电池|电池技术|固态电池"),
    (("新能源汽车", "整车制造"),
     r"车型|汽车|轿车|SUV|MPV|交付量|车企|造车|新车|上市|增程|混动|"
     r"充电桩|充电站|超充|补能"),
    (("新能源汽车", "自动驾驶技术"),
     r"自动驾驶|无人驾驶|辅助驾驶|智驾技术|端到端大模型|CVPR"),
]
PROVINCE_INVEST = re.compile(
    r"投资[一-龥]{2,3}(丨|：|:|，|,|\s|$)|投资(建厂|建基地|扩产|设厂)")

BRIEFING = re.compile(  # 多主题简报/行情专栏，无唯一答案，丢弃
    r"早参|早报|日报|周报|公告精选|盘前|盘后|要闻|快讯|盘点|合集|汇总|一览|"
    r"本周|今日.*［|多家公司|业绩前瞻|盘中|午盘|收盘|行情|一周投资|投资热点|"
    r"市场动态|资本速递")

# ----------------------------------------- L3 判定（优先级即平局规则）
L3_RULES = [
    # (类别, 标题模式, 例外/让位模式——命中例外则本条不算)
    ("财报与业绩", r"财报|季报|年报|营收|收入|利润|净利|业绩|亏损|毛利率", None),
    ("产能与扩张",
     r"建厂|投产|产能|基地|开工|扩建|工厂落地|制造基地" + "|" + PROVINCE_INVEST.pattern,
     None),
    ("法律与监管",
     r"诉讼|起诉|禁令|调查|罚款|反垄断|仲裁|违规|处罚|召回|落马|被查|双规|"
     r"违纪|职务犯罪|贪腐|通报处分|抄袭|侵权",
     None),
    ("人事与组织",
     r"出任|辞职|离任|任命|履新|加盟|卸任|裁员|高管|人事调整|组织调整|离职|换新|"
     r"高管变动", r"落马|被查|职务犯罪|贪腐"),
    ("专利与知识产权", r"专利|知识产权|商标", None),
    ("产品发布",
     r"发布|推出|上市|开售|首发|亮相|新品|发售|开卖|全新|新款|新机|正式开启",
     r"财报|季报|年报|营收|业绩|报告"),
    ("价格与销量", r"降价|涨价|价格|售价|提价|调价|官降|销量|交付|出货量|市场份额",
     r"发布|推出|上市|开售|首发|亮相|新品|发售|开卖|全新|新款|新机"),
    ("资本与并购",
     r"收购|并购|入股|投资|融资|增持|减持|IPO|募资|挂牌|股权|领投|退出|清仓",
     PROVINCE_INVEST.pattern),
    ("合作与联盟", r"战略合作|合作|签约|联手|联合|联盟|伙伴|携手", r"职务犯罪"),
    ("技术与研究",
     r"技术突破|研发|新技术|攻克|工艺|制程|实验室|论文|基座模型|技术进展|模型进展",
     None),
]
L4_RULES = {
    "产品发布": {
        "硬件": r"手机|手表|汽车|车型|轿车|SUV|MPV|芯片|电池|耳机|平板|无人机|"
              r"电视|相机|PC|笔记本|机型|硬件|手环|眼镜|机器人|穿戴|Watch|"
              r"Mate\s?\d|iPhone",
        "软件与服务": r"系统|OS|操作系统|大模型|模型|智能体|应用商店|[Aa]pp|软件|"
                  r"云服务|固件",
    },
    "法律与监管": {
        "诉讼纠纷": r"诉讼|起诉|仲裁|侵权|禁令|上诉|判决|和解|抄袭",
        "监管调查": r"调查|反垄断|罚款|处罚|监管|听证|违规|被查|落马|职务犯罪|贪腐",
    },
    "资本与并购": {
        "收购并购": r"收购|并购|合并|借壳|要约",
        "投资融资": r"投资|融资|入股|IPO|募资|增持|领投|减持|退出|清仓|出售",
    },
    "价格与销量": {
        "价格调整": r"降价|涨价|价格|售价|提价|调价|官降",
        "销量数据": r"销量|交付|出货量|市场份额",
    },
}


def content_domain(title, content):
    """三段式：标题档（平台/AI/云类）→ 正文档·标题 → 正文档·标题+正文前段。"""
    for (l1, l2), pat in DOMAIN_TITLE:
        if re.search(pat, title):
            return l1, l2
    for (l1, l2), pat in DOMAIN_BOTH:
        if re.search(pat, title):
            return l1, l2
    text = title + "　" + content[:500]
    for (l1, l2), pat in DOMAIN_BOTH:
        if re.search(pat, text):
            return l1, l2
    return None


def label_l3(title):
    hits = []
    for cat, pat, excl in L3_RULES:
        if re.search(pat, title) and not (excl and re.search(excl, title)):
            hits.append(cat)
    return hits[0] if len(hits) == 1 else (None if hits else "DROP_NOSIGNAL")


def label_l4(l3, title):
    if l3 not in L4_TREE:
        return ""
    rules = L4_RULES[l3]
    if l3 == "产品发布":  # 硬件优先：软硬同时命中取硬件，无信号默认硬件
        hw = bool(re.search(rules["硬件"], title))
        sw = bool(re.search(rules["软件与服务"], title))
        return "硬件" if hw or not sw else "软件与服务"
    hits = [k for k, pat in rules.items() if re.search(pat, title)]
    return hits[0] if len(hits) == 1 else None


def build_taxonomy():
    leaves = []
    for l1, l2s in L2_TREE.items():
        if l1 == "其他与跨界":  # 兜底类：单叶子，不放全交叉
            leaves.append({"id": f"L{len(leaves)+1:03d}",
                           "path": [l1, l2s[0], "其他事件"]})
            continue
        for l2 in l2s:
            for l3 in L3_LIST:
                if l3 in L4_TREE:
                    for l4 in L4_TREE[l3]:
                        leaves.append({"id": f"L{len(leaves)+1:03d}",
                                       "path": [l1, l2, l3, l4]})
                else:
                    leaves.append({"id": f"L{len(leaves)+1:03d}",
                                   "path": [l1, l2, l3]})
    return leaves


def main():
    leaves = build_taxonomy()
    nodes = {tuple(lf["path"][:i + 1]) for lf in leaves for i in range(len(lf["path"]))}
    taxonomy = {
        "l1": L1_LIST, "l2_tree": L2_TREE, "l2_desc": L2_DESC,
        "l3_list": L3_LIST, "l3_desc": L3_DESC, "l4_tree": L4_TREE,
        "leaves": leaves,
        "stats": {"leaf_count": len(leaves), "l1": len(L1_LIST),
                  "l2": sum(len(v) for v in L2_TREE.values()),
                  "l3": len(L3_LIST), "total_nodes": len(nodes) + 1},
    }
    (ROOT / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False,
                                                   indent=1))

    items, seen, drop = [], set(), Counter()
    for f in sorted(CORPUS.glob("*/*.json")):
        company = f.parent.name
        if company not in COMPANY_L2:
            continue
        for it in json.loads(f.read_text()).get("items", []):
            ch, title = it.get("content_hash"), it.get("title", "")
            content = it.get("content", "")
            if ch in seen:
                continue
            if it.get("language") != "zh" or len(content) < 500 or len(title) < 8:
                continue
            if BRIEFING.search(title):
                drop["简报"] += 1
                continue
            dom = content_domain(title, content)
            l1, l2 = dom or COMPANY_L2[company]
            l3 = label_l3(title)
            if l3 == "DROP_NOSIGNAL":
                drop["无事件信号"] += 1
                continue
            if not l3:
                drop["多信号两可"] += 1
                continue
            l4 = label_l4(l3, title)
            if l4 is None:
                drop["L4两可"] += 1
                continue
            seen.add(ch)
            items.append({"title": title, "content": content,
                          "company": company, "source": it.get("source_name", ""),
                          "published_at": it.get("published_at", ""),
                          "content_domain": bool(dom),
                          "gold_path": ([l1, l2, l3, l4] if l4 else [l1, l2, l3])})

    random.shuffle(items)
    (ROOT / "pool_all.json").write_text(json.dumps(
        {"n": len(items), "items": items}, ensure_ascii=False))
    cnt_l3, cnt_co, picked = Counter(), Counter(), []
    for it in items:
        l3, co = it["gold_path"][2], it["company"]
        if cnt_l3[l3] >= 40 or cnt_co[co] >= 15:
            continue
        cnt_l3[l3] += 1
        cnt_co[co] += 1
        picked.append(it)
        if len(picked) >= 300:
            break
    for i, it in enumerate(picked):
        it["id"] = f"T{i+1:03d}"
    (ROOT / "testset.json").write_text(json.dumps(
        {"n": len(picked), "items": picked}, ensure_ascii=False, indent=1))

    print(f"候选 {len(items)}，丢弃 {dict(drop)}，采样 {len(picked)}")
    print("L3 分布:", dict(cnt_l3))
    print("内容领域覆盖(非回退):", sum(1 for i in picked if i["content_domain"]),
          "/", len(picked))
    print("实际出现叶子:", len(set('/'.join(i['gold_path']) for i in picked)))
    print(f"体系: {len(leaves)} 叶子 / {taxonomy['stats']['total_nodes']} 节点")


if __name__ == "__main__":
    main()
