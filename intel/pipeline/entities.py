# -*- coding: utf-8 -*-
"""实体规范化（04 §5.3）：优先级 = 稳定业务 ID → 已确认别名 → 规范名匹配 → 新建。

规则式归一（NFKC、去空白、企业后缀剥离）覆盖"小米/小米集团"这类同实体异名；
新别名不因单次相似度直接全局合并（无训练分类器，保持保守）。
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata

from .. import observe
from ..util import det_uuid, now_iso

# 企业后缀剥离表（仅用于规范化键，不改显示名）
_SUFFIXES = ["股份有限公司", "有限责任公司", "集团有限公司", "有限公司", "集团公司",
             "集团", "控股公司", "控股", "科技公司", "技术有限公司", "技术公司",
             "汽车公司", "公司", "Inc", "Ltd", "Corp", "Co"]

# S2 修复：产品线/业务前缀剥离表（过分裂 R1 根因之一——"蔚来汽车"与"蔚来"、
# "快手科技"与"快手"、"小鹏汽车"与"小鹏" 抽取时绑定不同实体，结构化召回通道断裂）。
# 剥离条件：剥离后剩余主体 ≥2 字；仅当剥离前的键在库里无独立实体时才并入剥离后的
# 主体（保守：先查 exact，再查剥离合并——不强行合并已独立存在的实体）。
_BUSINESS_PREFIXES = ["汽车", "科技", "智能", "新能源", "数字", "云计算"]

_PUNCT = re.compile(r"[\s·•・'’" + "\u200b" + r"()（）【】\[\]-]+")


def norm_key(name: str) -> str:
    s = unicodedata.normalize("NFKC", str(name or "")).lower().strip()
    s = _PUNCT.sub("", s)
    changed = True
    while changed and len(s) > 2:
        changed = False
        for suf in _SUFFIXES:
            low = suf.lower()
            if s.endswith(low) and len(s) - len(low) >= 2:
                s = s[: -len(low)]
                changed = True
    # 业务前缀：以"主体+业务词"形态命名的（蔚来汽车/快手科技），键归一到主体。
    # 只在主体 ≥2 字且主体不含分隔语义时执行；norm_key 是纯函数，重放确定。
    for pre in _BUSINESS_PREFIXES:
        if s.endswith(pre) and len(s) - len(pre) >= 2 and not s[:-len(pre)].endswith(pre):
            s = s[: -len(pre)]
            break
    return s


_ENTITY_TYPES = {"company", "brand", "product", "person", "other"}


def resolve_entity(conn: sqlite3.Connection, name: str, type_hint: str | None = None) -> str | None:
    """名称 → entity_id（命中别名/规范键），否则新建实体。"""
    name = (name or "").strip()
    if not name or len(name) > 80:
        return None
    nk = norm_key(name)
    if len(nk) < 2:
        return None
    # 1) 规范名 / 别名命中
    row = conn.execute("SELECT entity_id FROM entity WHERE norm_key=? AND deleted_at IS NULL",
                       (nk,)).fetchone()
    if row:
        return row["entity_id"]
    row = conn.execute(
        "SELECT entity_id FROM entity_alias WHERE norm_key=? "
        "AND entity_id IN (SELECT entity_id FROM entity WHERE deleted_at IS NULL)",
        (nk,)).fetchone()
    if row:
        # 补登记该显示名为别名
        _add_alias(conn, row["entity_id"], name, nk)
        return row["entity_id"]
    # 2) 新建
    etype = type_hint if type_hint in _ENTITY_TYPES else _infer_type(name)
    entity_id = det_uuid("ent", etype, nk)
    conn.execute(
        "INSERT OR IGNORE INTO entity(entity_id, type, canonical_name, norm_key, created_at) "
        "VALUES (?,?,?,?,?)",
        (entity_id, etype, name, nk, now_iso()))
    _add_alias(conn, entity_id, name, nk)
    return entity_id


def _add_alias(conn, entity_id: str, alias: str, nk: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO entity_alias(alias_id, entity_id, alias, norm_key, created_at) "
        "VALUES (?,?,?,?,?)",
        (det_uuid("alias", entity_id, nk), entity_id, alias, nk, now_iso()))


def _infer_type(name: str) -> str:
    if re.search(r"(公司|集团|银行|基金|事务所|Inc|Corp|Ltd)$", name):
        return "company"
    if re.search(r"(CEO|董事长|总裁|创始人|总监|经理|博士|先生|女士)$", name):
        return "person"
    return "other"


def mention_entities(mention_row) -> list[str]:
    """读取提及上已解析的实体 ID 列表（actors + objects）。"""
    ids = []
    for field in ("actor_json", "object_json"):
        try:
            arr = json.loads(mention_row[field] or "[]")
        except json.JSONDecodeError:
            arr = []
        for a in arr:
            if isinstance(a, dict) and a.get("entity_id"):
                ids.append(a["entity_id"])
    return ids


def run(conn: sqlite3.Connection) -> dict:
    """为所有 valid 提及解析实体并回写 entity_id。"""
    rows = conn.execute(
        "SELECT m.mention_id, m.actor_json, m.object_json FROM event_mention m "
        "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
        "WHERE m.status='valid' "
        "ORDER BY COALESCE(dv.published_at, dv.created_at, m.created_at), "
        "m.document_version_id, m.local_id").fetchall()
    bound = 0
    for r in rows:
        for field in ("actor_json", "object_json"):
            arr = json.loads(r[field] or "[]")
            dirty = False
            for a in arr:
                if isinstance(a, dict) and a.get("name") and not a.get("entity_id"):
                    eid = resolve_entity(conn, a["name"], a.get("type"))
                    if eid:
                        a["entity_id"] = eid
                        bound += 1
                        dirty = True
            if dirty:  # 只在有变化时回写
                conn.execute("UPDATE event_mention SET " + field + "=? WHERE mention_id=?",
                             (json.dumps(arr, ensure_ascii=False), r["mention_id"]))
    conn.commit()
    n_entities = conn.execute("SELECT COUNT(*) n FROM entity WHERE deleted_at IS NULL").fetchone()["n"]
    observe.emit(conn, "entity", f"实体规范化完成: {len(rows)} 提及 / {bound} 处绑定 / "
                 f"{n_entities} 实体", kind="entity.stage_done",
                 data={"mentions": len(rows), "bindings": bound, "entities": n_entities})
    return {"mentions": len(rows), "bindings": bound, "entities": n_entities}
