# -*- coding: utf-8 -*-
"""实体层级种子表：母公司—子品牌/子公司的从属关系边（subsidiary_of / brand_of）。

来源：公开常识性企业结构（2024-2026 稳定事实），asserted_by='seed' 标注来源等级，
可被人工 CRUD 覆盖；这些边只表达组织归属，不参与同事件判别。
对应 Zep/E²RAG 的实体层组织与 E07 §7.3"图内容须可回溯"——seed 边不依赖语料，
因此单独标注并在 UI 上显示"种子/常识"徽标，与语料证据边区分。
"""
from __future__ import annotations

import sqlite3

from . import observe
from .util import det_uuid, now_iso

# (母实体规范名, 子实体规范名, 关系, 备注)
SEED_HIERARCHY = [
    ("小米集团", "小米汽车", "subsidiary_of", "小米集团造车子公司"),
    ("小米集团", "Redmi", "brand_of", "红米品牌"),
    ("小米集团", "小米手机", "brand_of", "手机产品线"),
    ("小米集团", "小米平板", "brand_of", "平板产品线"),
    ("华为", "问界", "brand_of", "华为智选车 AITO（赛力斯生产）"),
    ("华为", "鸿蒙智行", "brand_of", "华为汽车生态联盟"),
    ("华为", "荣耀", "related_to", "荣耀已独立（2020 剥离），保留历史关联边"),
    ("比亚迪", "腾势", "brand_of", "高端品牌"),
    ("比亚迪", "仰望", "brand_of", "豪华品牌"),
    ("比亚迪", "方程豹", "brand_of", "个性化品牌"),
    ("比亚迪", "王朝网", "brand_of", "销售网络"),
    ("比亚迪", "海洋网", "brand_of", "销售网络"),
    ("吉利控股", "吉利汽车", "subsidiary_of", "上市公司主体"),
    ("吉利汽车", "极氪", "subsidiary_of", "2024 并表、2025 私有化整合"),
    ("吉利汽车", "领克", "subsidiary_of", "合资品牌"),
    ("吉利汽车", "银河", "brand_of", "新能源系列"),
    ("蔚来", "乐道", "brand_of", "大众市场品牌"),
    ("蔚来", "萤火虫", "brand_of", "小车品牌"),
    ("蔚来", "NIO", "brand_of", "主品牌英文名"),
    ("长城汽车", "魏牌", "brand_of", "高端品牌"),
    ("长城汽车", "坦克", "brand_of", "越野品牌"),
    ("长城汽车", "欧拉", "brand_of", "电动车品牌"),
    ("长城汽车", "哈弗", "brand_of", "SUV 品牌"),
    ("上汽集团", "智己", "subsidiary_of", "高端电动品牌"),
    ("上汽集团", "荣威", "brand_of", "乘用车品牌"),
    ("上汽集团", "飞凡", "brand_of", "电动品牌"),
    ("上汽集团", "五菱", "subsidiary_of", "上汽通用五菱"),
    ("广汽集团", "埃安", "brand_of", "新能源品牌"),
    ("广汽集团", "昊铂", "brand_of", "高端品牌"),
    ("奇瑞汽车", "智界", "brand_of", "与华为合作品牌"),
    ("奇瑞汽车", "星途", "brand_of", "高端品牌"),
    ("小鹏汽车", "小鹏汇天", "subsidiary_of", "飞行汽车子公司"),
    ("理想汽车", "理想", "brand_of", "主品牌简称"),
    ("东风集团", "岚图", "brand_of", "高端电动品牌"),
    ("东风集团", "猛士", "brand_of", "越野电动品牌"),
    ("长安汽车", "深蓝", "brand_of", "新能源品牌"),
    ("长安汽车", "阿维塔", "subsidiary_of", "与华为宁德时代合资"),
    ("腾讯", "微信", "brand_of", "核心产品"),
    ("腾讯", "王者荣耀", "brand_of", "游戏产品"),
    ("腾讯", "混元", "brand_of", "大模型"),
    ("阿里巴巴", "淘宝", "brand_of", "电商"),
    ("阿里巴巴", "天猫", "brand_of", "电商"),
    ("阿里巴巴", "阿里云", "subsidiary_of", "云业务"),
    ("阿里巴巴", "通义", "brand_of", "大模型"),
    ("阿里巴巴", "菜鸟", "subsidiary_of", "物流"),
    ("阿里巴巴", "盒马", "subsidiary_of", "新零售"),
    ("字节跳动", "抖音", "brand_of", "短视频"),
    ("字节跳动", "今日头条", "brand_of", "资讯"),
    ("字节跳动", "豆包", "brand_of", "大模型/助手"),
    ("字节跳动", "飞书", "brand_of", "协同办公"),
    ("字节跳动", "火山引擎", "subsidiary_of", "云服务"),
    ("百度", "文心一言", "brand_of", "大模型"),
    ("百度", "萝卜快跑", "brand_of", "自动驾驶出行"),
    ("百度", "小度", "brand_of", "智能硬件"),
    ("京东", "京东物流", "subsidiary_of", "物流子公司"),
    ("京东", "京东健康", "subsidiary_of", "医疗健康子公司"),
    ("美团", "美团外卖", "brand_of", "外卖业务"),
    ("美团", "闪购", "brand_of", "即时零售"),
    ("网易", "逆水寒", "brand_of", "游戏产品"),
    ("网易", "有道", "subsidiary_of", "教育子公司"),
    ("OPPO", "一加", "brand_of", "高端品牌"),
    ("vivo", "iQOO", "brand_of", "性能品牌"),
    ("苹果", "iPhone", "brand_of", "手机产品线"),
    ("苹果", "Mac", "brand_of", "电脑产品线"),
    ("苹果", "iPad", "brand_of", "平板产品线"),
    ("苹果", "Vision Pro", "brand_of", "空间计算设备"),
    ("大疆", "Mavic", "brand_of", "无人机产品线"),
    ("宁德时代", "麒麟电池", "brand_of", "电池产品"),
    ("科大讯飞", "星火", "brand_of", "大模型"),
]

# 别名种子（同一实体的常见别名 → 规范名），提高实体消解召回
SEED_ALIASES = [
    ("小米集团", ["小米", "Xiaomi", "小米公司"]),
    ("小米汽车", ["Xiaomi EV", "小米SU7厂商"]),
    ("华为", ["华为技术", "Huawei"]),
    ("比亚迪", ["BYD"]),
    ("蔚来", ["NIO", "蔚来汽车"]),
    ("小鹏汽车", ["小鹏", "XPeng", "Xpeng"]),
    ("理想汽车", ["理想", "Li Auto"]),
    ("特斯拉", ["Tesla"]),
    ("腾讯", ["Tencent"]),
    ("阿里巴巴", ["阿里", "Alibaba"]),
    ("字节跳动", ["字节", "ByteDance"]),
    ("百度", ["Baidu"]),
    ("京东", ["JD", "JD.com"]),
    ("美团", ["Meituan"]),
    ("苹果", ["Apple", "苹果公司"]),
    ("OPPO", ["oppo"]),
    ("vivo", ["VIVO", "维沃"]),
    ("荣耀", ["Honor"]),
    ("大疆", ["DJI", "大疆创新"]),
    ("宁德时代", ["CATL"]),
    ("吉利汽车", ["吉利"]),
    ("长城汽车", ["长城"]),
    ("上汽集团", ["上汽"]),
    ("广汽集团", ["广汽"]),
    ("奇瑞汽车", ["奇瑞"]),
    ("网易", ["NetEase"]),
    ("科大讯飞", ["讯飞"]),
    ("中芯国际", ["SMIC"]),
    ("快手", ["Kuaishou"]),
    ("哔哩哔哩", ["B站", "bilibili"]),
    ("联想", ["Lenovo", "联想集团"]),
]


def apply_seeds(conn: sqlite3.Connection) -> dict:
    """写入层级边与别名种子（幂等）。返回统计。"""
    from .pipeline.entities import norm_key, resolve_entity
    edges = aliases = 0
    for parent, child, rel, note in SEED_HIERARCHY:
        pid = resolve_entity(conn, parent, "company")
        cid = resolve_entity(conn, child, "company")
        if not pid or not cid or pid == cid:
            continue
        rid = det_uuid("rel", "entity", pid, "entity", cid, rel)
        cur = conn.execute(
            "INSERT OR IGNORE INTO semantic_relation(relation_id, from_type, from_id, to_type, "
            "to_id, relation, asserted_by, evidence_json, note, created_by, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rid, "entity", cid, "entity", pid, rel, "seed", "[]", note, "seed", now_iso()))
        if cur.rowcount == 0:
            conn.execute("UPDATE semantic_relation SET deleted_at=NULL WHERE relation_id=?", (rid,))
        else:
            edges += 1
    for canonical, alias_list in SEED_ALIASES:
        eid = resolve_entity(conn, canonical, "company")
        if not eid:
            continue
        for alias in alias_list:
            nk = norm_key(alias)
            if not nk:
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO entity_alias(alias_id, entity_id, alias, norm_key, "
                "created_at) VALUES (?,?,?,?,?)",
                (det_uuid("alias", eid, nk), eid, alias, nk, now_iso()))
            aliases += cur.rowcount
    conn.commit()
    observe.emit(conn, "entity", f"层级种子: +{edges} 边 / +{aliases} 别名",
                 kind="entity.seeds", data={"edges": edges, "aliases": aliases})
    return {"edges": edges, "aliases": aliases}
