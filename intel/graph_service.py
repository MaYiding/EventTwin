# -*- coding: utf-8 -*-
"""图服务：节点/边增删改查 + 四类修复操作（04 §11.1）。

- 节点类型：entity / event(event_cluster) / process / assertion；
- 边（semantic_relation）：participates_in、part_of_process、precedes、follows、
  supports、contradicts、corrects、retracts、related_to；
- 修复操作：move_mention / split_cluster / merge_clusters / retract_assertion，
  均带 expected_version 乐观锁（冲突返回 409）与 outbox 派生失效；
- 图是可重建的派生层：所有人工操作同时写 resolution_decision 留痕。
"""
from __future__ import annotations

import json
import sqlite3

from . import observe
from .pipeline.assertions import canonical
from .pipeline.resolve import (_add_edge, _outbox_event, _refresh_cluster,
                               _update_precedes)
from .util import det_uuid, now_iso


class Conflict(Exception):
    """乐观锁版本冲突 → HTTP 409。"""


class NotFound(Exception):
    """对象不存在 → HTTP 404。"""


class BadRequest(Exception):
    """参数错误 → HTTP 400。"""


# ---------------------------------------------------------------------------
# 查（读路径）
# ---------------------------------------------------------------------------

NODE_FIELDS = {
    "entity": ("entity_id AS id", "type AS subtype", "canonical_name AS title",
               "created_at", "updated_at", "deleted_at"),
    "event": ("cluster_id AS id", "event_type AS subtype", "summary AS title",
              "event_time_lower", "event_time_upper", "version", "state", "process_id",
              "created_at", "updated_at", "deleted_at", "redirect_to"),
    "process": ("process_id AS id", "family AS subtype", "title", "created_at"),
    "assertion": ("assertion_id AS id", "predicate AS subtype", "slot_key AS title",
                  "value_json", "valid_from", "valid_to", "status", "correction_of",
                  "created_at"),
}


def list_nodes(conn, ntype: str, q: str | None = None, limit: int = 100,
               offset: int = 0) -> list[dict]:
    if ntype == "entity":
        # 连接度排序走 JOIN 预聚合：关联子查询会对每个实体全表扫 semantic_relation
        sql = ("SELECT e.entity_id AS id, e.type AS subtype, e.canonical_name AS title, "
               "e.created_at, e.updated_at, e.deleted_at, "
               "(SELECT COUNT(*) FROM entity_alias a WHERE a.entity_id=e.entity_id) AS aliases "
               "FROM entity e LEFT JOIN (SELECT from_id, COUNT(*) n "
               "FROM semantic_relation WHERE deleted_at IS NULL GROUP BY from_id) deg "
               "ON deg.from_id=e.entity_id WHERE 1=1 ")
        args: list = []
        if q:
            sql += "AND (e.canonical_name LIKE ? OR e.entity_id IN "
            sql += "(SELECT entity_id FROM entity_alias WHERE alias LIKE ?)) "
            args += [f"%{q}%", f"%{q}%"]
        sql += "ORDER BY COALESCE(deg.n, 0) DESC, e.canonical_name LIMIT ? OFFSET ?"
        args += [limit, offset]
    elif ntype == "event":
        sql = ("SELECT cluster_id AS id, event_type AS subtype, summary AS title, version, "
               "state, process_id, event_time_lower, event_time_upper, created_at, updated_at, "
               "deleted_at, redirect_to, "
               "(SELECT COUNT(*) FROM cluster_membership m WHERE m.cluster_id=event_cluster.cluster_id "
               "AND m.removed_at IS NULL) AS members FROM event_cluster WHERE 1=1 ")
        args = []
        if q:
            sql += "AND (summary LIKE ? OR event_type LIKE ?) "
            args += [f"%{q}%", f"%{q}%"]
        sql += "ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
    elif ntype == "process":
        sql = ("SELECT p.process_id AS id, p.family AS subtype, p.title, p.created_at, "
               "(SELECT COUNT(*) FROM event_cluster c WHERE c.process_id=p.process_id "
               "AND c.deleted_at IS NULL) AS members FROM process p ")
        args = []
        if q:
            sql += "WHERE p.title LIKE ? "
            args.append(f"%{q}%")
        sql += "ORDER BY p.created_at LIMIT ? OFFSET ?"
        args += [limit, offset]
    elif ntype == "assertion":
        sql = ("SELECT assertion_id AS id, predicate AS subtype, slot_key AS title, value_json, "
               "valid_from, valid_to, status, correction_of, retraction_of, event_id, "
               "created_at FROM assertion ")
        args = []
        if q:
            sql += "WHERE (slot_key LIKE ? OR value_json LIKE ?) "
            args += [f"%{q}%", f"%{q}%"]
        sql += "ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
    else:
        raise BadRequest(f"未知节点类型: {ntype}")
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def get_entity(conn, entity_id: str) -> dict:
    r = conn.execute("SELECT * FROM entity WHERE entity_id=?", (entity_id,)).fetchone()
    if r is None or r["deleted_at"]:
        raise NotFound("实体不存在（或已删除）")
    out = dict(r)
    out["aliases"] = [dict(a) for a in conn.execute(
        "SELECT alias_id, alias FROM entity_alias WHERE entity_id=?", (entity_id,)).fetchall()]
    out["events"] = [dict(x) for x in conn.execute(
        "SELECT c.cluster_id, c.summary, c.event_type, c.state FROM semantic_relation sr "
        "JOIN event_cluster c ON c.cluster_id=sr.to_id "
        "WHERE sr.relation='participates_in' AND sr.from_id=? AND sr.deleted_at IS NULL "
        "AND c.deleted_at IS NULL", (entity_id,)).fetchall()]
    out["facts"] = facts_for_subject(conn, entity_id)
    return out


def get_event(conn, cluster_id: str, version: int | None = None) -> dict:
    r = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=?", (cluster_id,)).fetchone()
    if r is None:
        raise NotFound("事件不存在")
    out = dict(r)
    out.pop("centroid", None)  # BLOB 不可 JSON 序列化；质心属派生投影层
    if version:
        snap = conn.execute(
            "SELECT * FROM cluster_version WHERE cluster_id=? AND version=?",
            (cluster_id, version)).fetchone()
        if snap is None:
            raise NotFound(f"版本 {version} 不存在")
        out["snapshot"] = json.loads(snap["snapshot_json"])
    out["versions"] = [v["version"] for v in conn.execute(
        "SELECT version FROM cluster_version WHERE cluster_id=? ORDER BY version",
        (cluster_id,)).fetchall()]
    out["members"] = []
    for m in conn.execute(
            "SELECT m.mention_id, m.frame_text, m.event_type, m.created_at, cm.added_at, "
            "dv.title AS doc_title, d.canonical_url, s.name AS source_name, sl.lineage_group "
            "FROM cluster_membership cm JOIN event_mention m ON m.mention_id=cm.mention_id "
            "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
            "LEFT JOIN document d ON d.document_id=dv.document_id "
            "LEFT JOIN source s ON s.source_id=d.source_id "
            "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
            "WHERE cm.cluster_id=? AND cm.removed_at IS NULL "
            "ORDER BY COALESCE(dv.published_at, cm.added_at)", (cluster_id,)).fetchall():
        row = dict(m)
        row["evidence"] = json.loads(
            conn.execute("SELECT evidence_json FROM event_mention WHERE mention_id=?",
                         (m["mention_id"],)).fetchone()["evidence_json"] or "[]")
        out["members"].append(row)
    out["entities"] = [dict(x) for x in conn.execute(
        "SELECT e.entity_id, e.canonical_name, e.type FROM semantic_relation sr "
        "JOIN entity e ON e.entity_id=sr.from_id "
        "WHERE sr.relation='participates_in' AND sr.to_id=? AND sr.deleted_at IS NULL "
        "AND e.deleted_at IS NULL", (cluster_id,)).fetchall()]
    out["assertions"] = [dict(x) for x in conn.execute(
        "SELECT assertion_id, predicate, value_json, valid_from, valid_to, status, "
        "correction_of FROM assertion WHERE event_id=?", (cluster_id,)).fetchall()]
    return out


def list_edges(conn, *, from_id: str | None = None, to_id: str | None = None,
               relation: str | None = None, limit: int = 200, offset: int = 0,
               include_deleted: bool = False) -> list[dict]:
    sql = "SELECT * FROM semantic_relation WHERE 1=1 "
    args: list = []
    if not include_deleted:
        sql += "AND deleted_at IS NULL "
    if from_id:
        sql += "AND from_id=? "
        args.append(from_id)
    if to_id:
        sql += "AND to_id=? "
        args.append(to_id)
    if relation:
        sql += "AND relation=? "
        args.append(relation)
    sql += "ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def graph_aggregate(conn) -> dict:
    """大图聚合视图（3000+ 事件时的默认态）：

    企业（canonical 公司实体）为超节点，携带事件数/类型分布/时间范围统计；
    企业间的边 = 共享子品牌层级或共同参与事件数。点击企业后前端切 focus 模式下钻。
    """
    # 两遍全库聚合替代逐实体 N+1（2670 实体 × 2 次全表扫 → 固定 2 条查询）
    stat_rows = conn.execute(
        "SELECT r.from_id AS fid, COUNT(DISTINCT c.cluster_id) n, "
        "MIN(c.event_time_lower) tmin, MAX(c.event_time_lower) tmax "
        "FROM semantic_relation r JOIN event_cluster c ON c.cluster_id=r.to_id "
        "WHERE r.relation='participates_in' AND r.deleted_at IS NULL "
        "AND c.deleted_at IS NULL GROUP BY r.from_id").fetchall()
    types_by_fid: dict[str, list] = {}
    for t in conn.execute(
            "SELECT r.from_id AS fid, c.event_type, COUNT(*) n "
            "FROM semantic_relation r JOIN event_cluster c ON c.cluster_id=r.to_id "
            "WHERE r.relation='participates_in' AND r.deleted_at IS NULL "
            "AND c.deleted_at IS NULL GROUP BY r.from_id, c.event_type").fetchall():
        types_by_fid.setdefault(t["fid"], []).append((t["event_type"], t["n"]))
    companies = {c["entity_id"]: c for c in conn.execute(
        "SELECT entity_id, canonical_name, type FROM entity "
        "WHERE deleted_at IS NULL AND type IN ('company','brand')").fetchall()}
    nodes = []
    for s in stat_rows:
        c = companies.get(s["fid"])
        if not c or not s["n"]:
            continue
        top_types = sorted(types_by_fid.get(s["fid"], []), key=lambda x: -x[1])[:3]
        nodes.append({"id": "entity:" + c["entity_id"], "raw_id": c["entity_id"],
                      "type": "entity", "subtype": c["type"],
                      "label": c["canonical_name"],
                      "n_events": s["n"], "tmin": (s["tmin"] or "")[:10],
                      "tmax": (s["tmax"] or "")[:10],
                      "types": dict(top_types)})
    # 总览规模控制：长尾碎片实体太多会拖垮前端力导向渲染；
    # 保留事件数 Top 80 企业（30 家订阅企业全覆盖），其余走 focus 下钻查看
    nodes.sort(key=lambda n: -n["n_events"])
    if len(nodes) > 80:
        nodes = nodes[:80]
    edges = []
    # 企业间层级边（seed）
    ids = {n["raw_id"] for n in nodes}
    for r in conn.execute(
            "SELECT from_id, to_id, relation FROM semantic_relation WHERE relation IN "
            "('subsidiary_of','brand_of') AND deleted_at IS NULL").fetchall():
        if r["from_id"] in ids and r["to_id"] in ids:
            edges.append({"id": f"{r['from_id']}->{r['to_id']}", "source": "entity:" + r["from_id"],
                          "target": "entity:" + r["to_id"], "relation": r["relation"],
                          "asserted_by": "seed"})
    return {"mode": "aggregate", "nodes": nodes, "edges": edges}


def graph_view(conn, *, focus: str | None = None, hops: int = 1, limit: int = 200,
               mode: str | None = None) -> dict:
    """给前端图谱视图的节点+边打包（默认全图采样，focus 时局部扩展 ≤2 跳）。"""
    if mode == "aggregate":
        return graph_aggregate(conn)
    hops = min(hops, 2)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    # 全量载入非删除边（6.5 万级，内存 BFS；旧的 limit=2000 截断会让 focus 扩展漏掉绝大多数邻接边）
    all_edges = [e for e in list_edges(conn, limit=500000) if not e["deleted_at"]]
    if focus:
        frontier = {focus}
        seen = set()
        for _ in range(hops):
            nxt = set()
            for e in all_edges:
                if e["from_id"] in frontier and e["to_id"] not in seen:
                    nxt.add(e["to_id"])
                if e["to_id"] in frontier and e["from_id"] not in seen:
                    nxt.add(e["from_id"])
            seen |= frontier
            frontier = nxt - seen
            if not frontier:
                break
        seen |= frontier
        all_edges = [e for e in all_edges if e["from_id"] in seen or e["to_id"] in seen]
        # 规模上限：枢纽实体 2 跳可达上千节点，按子图内度数截断，防止前端力导向渲染冻结
        max_nodes = 500
        endpoints = {e["from_id"] for e in all_edges} | {e["to_id"] for e in all_edges}
        if len(endpoints) > max_nodes:
            deg: dict[str, int] = {}
            for e in all_edges:
                deg[e["from_id"]] = deg.get(e["from_id"], 0) + 1
                deg[e["to_id"]] = deg.get(e["to_id"], 0) + 1
            keep = set(sorted(deg, key=lambda x: -deg[x])[:max_nodes - 1]) | {focus}
            all_edges = [e for e in all_edges
                         if e["from_id"] in keep and e["to_id"] in keep]
    else:
        # 全图：按重要性采样（关系数多的节点优先）
        deg: dict[str, int] = {}
        for e in all_edges:
            deg[e["from_id"]] = deg.get(e["from_id"], 0) + 1
            deg[e["to_id"]] = deg.get(e["to_id"], 0) + 1
        ids = set(sorted(deg, key=lambda x: -deg[x])[:limit])
        all_edges = [e for e in all_edges if e["from_id"] in ids and e["to_id"] in ids]

    def _node(ntype, nid):
        key = f"{ntype}:{nid}"
        if key in nodes:
            return
        if ntype == "entity":
            r = conn.execute("SELECT entity_id, type, canonical_name FROM entity WHERE entity_id=?",
                             (nid,)).fetchone()
            if r:
                nodes[key] = {"id": key, "raw_id": nid, "type": "entity", "subtype": r["type"],
                              "label": r["canonical_name"]}
        elif ntype == "event":
            r = conn.execute("SELECT cluster_id, event_type, summary, state FROM event_cluster "
                             "WHERE cluster_id=?", (nid,)).fetchone()
            if r:
                nodes[key] = {"id": key, "raw_id": nid, "type": "event",
                              "subtype": r["event_type"], "label": (r["summary"] or "")[:40],
                              "state": r["state"]}
        elif ntype == "process":
            r = conn.execute("SELECT process_id, family, title FROM process WHERE process_id=?",
                             (nid,)).fetchone()
            if r:
                nodes[key] = {"id": key, "raw_id": nid, "type": "process",
                              "subtype": r["family"], "label": r["title"]}
        elif ntype == "assertion":
            r = conn.execute("SELECT assertion_id, predicate, value_json FROM assertion "
                             "WHERE assertion_id=?", (nid,)).fetchone()
            if r:
                val = json.loads(r["value_json"]).get("value", "")
                nodes[key] = {"id": key, "raw_id": nid, "type": "assertion",
                              "subtype": r["predicate"], "label": f"{r['predicate']}={val}"[:30]}

    for e in all_edges:
        _node(e["from_type"], e["from_id"])
        _node(e["to_type"], e["to_id"])
        edges.append({"id": e["relation_id"], "source": f"{e['from_type']}:{e['from_id']}",
                      "target": f"{e['to_type']}:{e['to_id']}", "relation": e["relation"],
                      "asserted_by": e["asserted_by"]})
    return {"nodes": list(nodes.values()), "edges": edges}


def facts_for_subject(conn, entity_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT DISTINCT slot_key FROM assertion WHERE subject_entity_id=?", (entity_id,)).fetchall()
    from .query_service import current_fact
    return [current_fact(conn, r["slot_key"]) for r in rows]


# ---------------------------------------------------------------------------
# 增 / 改 / 删（写路径，全部留痕 + outbox）
# ---------------------------------------------------------------------------

def create_entity(conn, name: str, subtype: str = "other", aliases: list[str] | None = None,
                  note: str | None = None) -> dict:
    from .pipeline.entities import norm_key, resolve_entity
    if not name or not name.strip():
        raise BadRequest("实体名不能为空")
    existing = conn.execute("SELECT entity_id, deleted_at FROM entity WHERE norm_key=?",
                            (norm_key(name),)).fetchone()
    if existing and not existing["deleted_at"]:
        raise BadRequest(f"实体已存在: {existing['entity_id']}")
    eid = resolve_entity(conn, name.strip(), subtype)
    if existing and existing["deleted_at"]:
        # 软删除过的同名实体 → 复活（清 deleted_at、恢复关系边）
        conn.execute("UPDATE entity SET deleted_at=NULL, updated_at=? WHERE entity_id=?",
                     (now_iso(), eid))
        conn.execute("UPDATE semantic_relation SET deleted_at=NULL WHERE "
                     "(from_type='entity' AND from_id=?) OR (to_type='entity' AND to_id=?)",
                     (eid, eid))
    for a in aliases or []:
        nk = norm_key(a)
        if nk and nk != norm_key(name):
            conn.execute(
                "INSERT OR IGNORE INTO entity_alias(alias_id, entity_id, alias, norm_key, created_at)"
                " VALUES (?,?,?,?,?)", (det_uuid("alias", eid, nk), eid, a, nk, now_iso()))
    if note:
        conn.execute("UPDATE entity SET note=? WHERE entity_id=?", (note, eid))
    conn.commit()
    observe.emit(conn, "crud", f"新建实体: {name}", kind="crud.entity.create", target_id=eid,
                 data={"name": name, "type": subtype, "aliases": aliases or [],
                       "revived": bool(existing and existing["deleted_at"])})
    return get_entity(conn, eid)


def update_entity(conn, entity_id: str, *, name: str | None = None,
                  add_aliases: list[str] | None = None, remove_alias_ids: list[str] | None = None,
                  note: str | None = None, expected_updated: str | None = None) -> dict:
    r = conn.execute("SELECT * FROM entity WHERE entity_id=? AND deleted_at IS NULL",
                     (entity_id,)).fetchone()
    if r is None:
        raise NotFound("实体不存在")
    if expected_updated is not None and r["updated_at"] != expected_updated:
        raise Conflict("实体已被他人修改，请刷新后重试")
    from .pipeline.entities import norm_key
    if name and name.strip() and name != r["canonical_name"]:
        dup = conn.execute("SELECT entity_id FROM entity WHERE norm_key=? AND entity_id<>?",
                           (norm_key(name), entity_id)).fetchone()
        if dup:
            raise BadRequest("规范名与其他实体冲突")
        conn.execute("UPDATE entity SET canonical_name=? WHERE entity_id=?", (name, entity_id))
    for a in add_aliases or []:
        nk = norm_key(a)
        if nk:
            conn.execute(
                "INSERT OR IGNORE INTO entity_alias(alias_id, entity_id, alias, norm_key, "
                "created_at) VALUES (?,?,?,?,?)",
                (det_uuid("alias", entity_id, nk), entity_id, a, nk, now_iso()))
    for aid in remove_alias_ids or []:
        conn.execute("DELETE FROM entity_alias WHERE alias_id=? AND entity_id=?", (aid, entity_id))
    if note is not None:
        conn.execute("UPDATE entity SET note=? WHERE entity_id=?", (note, entity_id))
    conn.execute("UPDATE entity SET updated_at=? WHERE entity_id=?", (now_iso(), entity_id))
    conn.commit()
    observe.emit(conn, "crud", f"更新实体: {r['canonical_name']}", kind="crud.entity.update",
                 target_id=entity_id)
    return get_entity(conn, entity_id)


def delete_entity(conn, entity_id: str, *, expected_updated: str | None = None) -> dict:
    r = conn.execute("SELECT * FROM entity WHERE entity_id=? AND deleted_at IS NULL",
                     (entity_id,)).fetchone()
    if r is None:
        raise NotFound("实体不存在")
    if expected_updated is not None and (r["updated_at"] or "") != expected_updated:
        raise Conflict("实体已被他人修改")
    n_events = conn.execute(
        "SELECT COUNT(*) n FROM semantic_relation WHERE relation='participates_in' "
        "AND from_id=? AND deleted_at IS NULL", (entity_id,)).fetchone()["n"]
    if n_events:
        raise BadRequest(f"实体仍参与 {n_events} 个事件，请先处理事件关联（或使用强制）")
    conn.execute("UPDATE entity SET deleted_at=? WHERE entity_id=?", (now_iso(), entity_id))
    conn.execute("UPDATE semantic_relation SET deleted_at=? WHERE from_id=? OR to_id=?",
                 (now_iso(), entity_id, entity_id))
    conn.commit()
    observe.emit(conn, "crud", f"删除实体(软): {r['canonical_name']}", level="warn",
                 kind="crud.entity.delete", target_id=entity_id)
    return {"deleted": entity_id}


def create_edge(conn, from_type: str, from_id: str, to_type: str, to_id: str,
                relation: str, *, note: str | None = None,
                evidence: list | None = None) -> dict:
    _check_node(conn, from_type, from_id)
    _check_node(conn, to_type, to_id)
    rid = _add_edge(conn, from_type, from_id, to_type, to_id, relation,
                    asserted_by="manual", note=note, evidence=evidence)
    if rid is None:
        raise BadRequest("边已存在或端点相同")
    conn.commit()
    observe.emit(conn, "crud", f"新建边: {relation} {from_id[:10]}→{to_id[:10]}",
                 kind="crud.edge.create", target_id=rid, data={"relation": relation})
    row = conn.execute("SELECT * FROM semantic_relation WHERE relation_id=?", (rid,)).fetchone()
    return dict(row)


def delete_edge(conn, relation_id: str) -> dict:
    r = conn.execute("SELECT * FROM semantic_relation WHERE relation_id=? AND deleted_at IS NULL",
                     (relation_id,)).fetchone()
    if r is None:
        raise NotFound("边不存在")
    conn.execute("UPDATE semantic_relation SET deleted_at=? WHERE relation_id=?",
                 (now_iso(), relation_id))
    conn.commit()
    observe.emit(conn, "crud", f"删除边: {r['relation']} {r['from_id'][:10]}→{r['to_id'][:10]}",
                 level="warn", kind="crud.edge.delete", target_id=relation_id)
    return {"deleted": relation_id}


def update_edge(conn, relation_id: str, *, note: str | None = None,
                valid_from: str | None = None, valid_to: str | None = None) -> dict:
    r = conn.execute("SELECT * FROM semantic_relation WHERE relation_id=? AND deleted_at IS NULL",
                     (relation_id,)).fetchone()
    if r is None:
        raise NotFound("边不存在")
    conn.execute(
        "UPDATE semantic_relation SET note=COALESCE(?, note), valid_from=COALESCE(?, valid_from), "
        "valid_to=COALESCE(?, valid_to) WHERE relation_id=?",
        (note, valid_from, valid_to, relation_id))
    conn.commit()
    observe.emit(conn, "crud", f"更新边: {relation_id[:10]}", kind="crud.edge.update",
                 target_id=relation_id)
    return dict(conn.execute("SELECT * FROM semantic_relation WHERE relation_id=?",
                             (relation_id,)).fetchone())


def _check_node(conn, ntype: str, nid: str) -> None:
    table = {"entity": "entity", "event": "event_cluster", "process": "process",
             "assertion": "assertion", "evidence": None}.get(ntype)
    if table is None:
        if ntype == "evidence":
            return
        raise BadRequest(f"未知节点类型: {ntype}")
    col = {"entity": "entity_id", "event": "cluster_id", "process": "process_id",
           "assertion": "assertion_id"}[ntype]
    row = conn.execute(f"SELECT {col} FROM {table} WHERE {col}=?", (nid,)).fetchone()
    if row is None:
        raise NotFound(f"{ntype} 不存在: {nid}")


# ---------------------------------------------------------------------------
# 人工事件（增/改/删）
# ---------------------------------------------------------------------------

def create_event(conn, *, event_type: str, title: str, event_time_lower: str | None = None,
                 event_time_upper: str | None = None, entity_ids: list[str] | None = None,
                 process_id: str | None = None, note: str | None = None) -> dict:
    cluster_id = det_uuid("cluster-manual", title, event_type, event_time_lower or "")
    now = now_iso()
    # 同 ID 事件已被软删除 → 复活并升版本
    prev = conn.execute("SELECT version, deleted_at FROM event_cluster WHERE cluster_id=?",
                        (cluster_id,)).fetchone()
    if prev and prev["deleted_at"]:
        conn.execute(
            "UPDATE event_cluster SET deleted_at=NULL, state='resolved', "
            "updated_at=? WHERE cluster_id=?", (now, cluster_id))
        _outbox_event(conn, cluster_id, prev["version"], "event.changed",
                      {"cluster_id": cluster_id, "version": prev["version"], "revived": True})
    decision_id = det_uuid("dec-manual", cluster_id, str(now))
    summary = f"[人工] {title}"
    if prev is None:
        conn.execute(
            "INSERT OR IGNORE INTO event_cluster(cluster_id, version, event_type, state, "
            "process_id, frame_json, event_time_lower, event_time_upper, first_seen, last_seen, "
            "summary, centroid, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cluster_id, 1, event_type, "resolved", process_id,
             json.dumps({"frame_text": summary, "scope": {}, "phase": "manual",
                         "representatives": [], "note": note}, ensure_ascii=False),
             event_time_lower, event_time_upper, now, now, summary, None, now, now))
    conn.execute(
        "INSERT OR REPLACE INTO cluster_version(cluster_id, version, decision_id, snapshot_json, "
        "recorded_at) VALUES (?,?,?,?,?)",
        (cluster_id, 1, decision_id,
         json.dumps({"members": [], "summary": summary, "state": "resolved",
                     "manual": True}, ensure_ascii=False), now))
    conn.execute(
        "INSERT OR REPLACE INTO resolution_decision(decision_id, mention_id, action, "
        "target_cluster_id, reason_code, policy_version, created_at) VALUES (?,?,?,?,?,?,?)",
        (decision_id, f"manual:{cluster_id}", "manual", cluster_id, "人工创建事件",
         "manual-v1", now))
    for eid in entity_ids or []:
        _check_node(conn, "entity", eid)
        _add_edge(conn, "entity", eid, "event", cluster_id, "participates_in",
                  asserted_by="manual", note="人工建边")
    if process_id:
        _check_node(conn, "process", process_id)
        _add_edge(conn, "event", cluster_id, "process", process_id, "part_of_process",
                  asserted_by="manual", note="人工挂载过程")
    _outbox_event(conn, cluster_id, 1, "event.changed", {"cluster_id": cluster_id, "version": 1})
    conn.commit()
    observe.emit(conn, "crud", f"新建事件(人工): {title[:50]}", kind="crud.event.create",
                 target_id=cluster_id, data={"type": event_type, "entities": entity_ids or []})
    return get_event(conn, cluster_id)


def update_event(conn, cluster_id: str, *, expected_version: int, event_type: str | None = None,
                 state: str | None = None, summary: str | None = None,
                 event_time_lower: str | None = None, event_time_upper: str | None = None,
                 note: str | None = None) -> dict:
    r = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                     (cluster_id,)).fetchone()
    if r is None:
        raise NotFound("事件不存在")
    if expected_version != r["version"]:
        raise Conflict(f"版本冲突: 期望 {expected_version} 实际 {r['version']}")
    frame = json.loads(r["frame_json"] or "{}")
    if note is not None:
        frame["note"] = note
    new_version = r["version"] + 1
    conn.execute(
        "UPDATE event_cluster SET version=?, event_type=COALESCE(?, event_type), "
        "state=COALESCE(?, state), summary=COALESCE(?, summary), "
        "event_time_lower=COALESCE(?, event_time_lower), "
        "event_time_upper=COALESCE(?, event_time_upper), frame_json=?, updated_at=? "
        "WHERE cluster_id=?",
        (new_version, event_type, state, summary, event_time_lower, event_time_upper,
         json.dumps(frame, ensure_ascii=False), now_iso(), cluster_id))
    decision_id = det_uuid("dec-manual-update", cluster_id, str(new_version))
    conn.execute(
        "INSERT INTO resolution_decision(decision_id, mention_id, action, target_cluster_id, "
        "reason_code, policy_version, created_at) VALUES (?,?,?,?,?,?,?)",
        (decision_id, f"manual:{cluster_id}", "manual", cluster_id, "人工更新事件字段",
         "manual-v1", now_iso()))
    conn.execute(
        "INSERT INTO cluster_version(cluster_id, version, decision_id, snapshot_json, recorded_at) "
        "VALUES (?,?,?,?,?)",
        (cluster_id, new_version, decision_id,
         json.dumps({"members": [], "summary": summary or r["summary"],
                     "state": state or r["state"], "manual_update": True}, ensure_ascii=False),
         now_iso()))
    _outbox_event(conn, cluster_id, new_version, "event.changed",
                  {"cluster_id": cluster_id, "version": new_version, "manual": True})
    conn.commit()
    observe.emit(conn, "crud", f"更新事件: {cluster_id[:12]} → v{new_version}",
                 kind="crud.event.update", target_id=cluster_id)
    return get_event(conn, cluster_id)


def delete_event(conn, cluster_id: str, *, expected_version: int, force: bool = False) -> dict:
    r = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                     (cluster_id,)).fetchone()
    if r is None:
        raise NotFound("事件不存在")
    if expected_version != r["version"]:
        raise Conflict(f"版本冲突: 期望 {expected_version} 实际 {r['version']}")
    n_members = conn.execute(
        "SELECT COUNT(*) n FROM cluster_membership WHERE cluster_id=? AND removed_at IS NULL",
        (cluster_id,)).fetchone()["n"]
    if n_members and not force:
        raise BadRequest(f"事件仍有 {n_members} 个成员提及，需 force 或先移动成员")
    now = now_iso()
    conn.execute(
        "UPDATE cluster_membership SET removed_at=?, removed_reason='事件删除' "
        "WHERE cluster_id=? AND removed_at IS NULL", (now, cluster_id))
    conn.execute("UPDATE event_cluster SET deleted_at=?, state='deleted', updated_at=? "
                 "WHERE cluster_id=?", (now, now, cluster_id))
    conn.execute("UPDATE semantic_relation SET deleted_at=? WHERE "
                 "(from_type='event' AND from_id=?) OR (to_type='event' AND to_id=?)",
                 (now, cluster_id, cluster_id))
    new_version = r["version"] + 1
    _outbox_event(conn, cluster_id, new_version, "event.deleted",
                  {"cluster_id": cluster_id, "version": new_version})
    conn.commit()
    observe.emit(conn, "crud", f"删除事件(软): {cluster_id[:12]}（成员 {n_members}）",
                 level="warn", kind="crud.event.delete", target_id=cluster_id)
    return {"deleted": cluster_id, "members_closed": n_members}


# ---------------------------------------------------------------------------
# 四类修复操作（04 §11.1）
# ---------------------------------------------------------------------------

def move_mention(conn, mention_id: str, target_cluster_id: str, *, reason: str,
                 expected_source_version: int | None = None,
                 expected_target_version: int | None = None) -> dict:
    cur = conn.execute(
        "SELECT cm.*, c.version FROM cluster_membership cm "
        "JOIN event_cluster c ON c.cluster_id=cm.cluster_id "
        "WHERE cm.mention_id=? AND cm.removed_at IS NULL", (mention_id,)).fetchone()
    if cur is None:
        raise NotFound("提及不存在活跃归属")
    target = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                          (target_cluster_id,)).fetchone()
    if target is None:
        raise NotFound("目标事件不存在")
    if expected_source_version is not None and cur["version"] != expected_source_version:
        raise Conflict(f"源簇版本冲突: 期望 {expected_source_version} 实际 {cur['version']}")
    if expected_target_version is not None and target["version"] != expected_target_version:
        raise Conflict(f"目标簇版本冲突: 期望 {expected_target_version} 实际 {target['version']}")
    now = now_iso()
    decision_id = det_uuid("dec-move", mention_id, target_cluster_id, now)
    conn.execute("UPDATE cluster_membership SET removed_at=?, removed_reason=? "
                 "WHERE membership_id=?", (now, f"移出: {reason}", cur["membership_id"]))
    conn.execute(
        "INSERT OR IGNORE INTO cluster_membership(membership_id, mention_id, cluster_id, "
        "decision_id, added_at) VALUES (?,?,?,?,?)",
        (det_uuid("mem", mention_id, target_cluster_id, "move"), mention_id,
         target_cluster_id, decision_id, now))
    conn.execute(
        "INSERT INTO resolution_decision(decision_id, mention_id, action, target_cluster_id, "
        "reason_code, policy_version, created_at) VALUES (?,?,?,?,?,?,?)",
        (decision_id, mention_id, "manual", target_cluster_id, f"人工移动成员: {reason}",
         "manual-v1", now))
    conn.commit()
    _refresh_cluster(conn, cur["cluster_id"])
    refreshed = _refresh_cluster(conn, target_cluster_id)
    conn.commit()
    observe.emit(conn, "crud", f"移动提及: {mention_id[:10]} {cur['cluster_id'][:10]}→"
                 f"{target_cluster_id[:10]}（{reason}）", level="warn",
                 kind="crud.op.move_mention", target_id=mention_id,
                 data={"from": cur["cluster_id"], "to": target_cluster_id, "reason": reason})
    return {"moved": mention_id, "from": cur["cluster_id"], "to": target_cluster_id,
            "target_version": refreshed["version"]}


def split_cluster(conn, cluster_id: str, mention_ids: list[str], *, reason: str,
                  expected_version: int) -> dict:
    r = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                     (cluster_id,)).fetchone()
    if r is None:
        raise NotFound("事件不存在")
    if expected_version != r["version"]:
        raise Conflict(f"版本冲突: 期望 {expected_version} 实际 {r['version']}")
    members = conn.execute(
        "SELECT mention_id FROM cluster_membership WHERE cluster_id=? AND removed_at IS NULL",
        (cluster_id,)).fetchall()
    member_set = {m["mention_id"] for m in members}
    move_set = set(mention_ids)
    if not move_set or not move_set <= member_set:
        raise BadRequest("拆分成员必须是当前簇的活跃成员")
    if move_set == member_set:
        raise BadRequest("不能把全部成员拆走（请改用 merge 或删除）")
    now = now_iso()
    new_id = det_uuid("cluster-split", cluster_id, canonical(sorted(move_set)))
    first = conn.execute("SELECT * FROM event_mention WHERE mention_id=?", (mention_ids[0],)).fetchone()
    conn.execute(
        "INSERT INTO event_cluster(cluster_id, version, event_type, state, process_id, "
        "frame_json, event_time_lower, event_time_upper, first_seen, last_seen, summary, "
        "centroid, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (new_id, 1, first["event_type"], "provisional", r["process_id"],
         json.dumps({"frame_text": first["frame_text"],
                     "scope": json.loads(first["scope_json"] or "{}"),
                     "split_from": cluster_id}, ensure_ascii=False),
         first["event_time_lower"], first["event_time_upper"], now, now,
         first["frame_text"], None, now, now))
    decision_id = det_uuid("dec-split", new_id)
    conn.execute(
        "INSERT INTO resolution_decision(decision_id, mention_id, action, target_cluster_id, "
        "reason_code, policy_version, created_at) VALUES (?,?,?,?,?,?,?)",
        (decision_id, f"split:{new_id}", "manual", new_id, f"人工拆簇: {reason}",
         "manual-v1", now))
    conn.execute(
        "INSERT INTO cluster_version(cluster_id, version, decision_id, snapshot_json, recorded_at) "
        "VALUES (?,?,?,?,?)",
        (new_id, 1, decision_id, json.dumps({"members": sorted(move_set), "split_from": cluster_id,
                                             "reason": reason}, ensure_ascii=False), now))
    for mid in move_set:
        conn.execute(
            "UPDATE cluster_membership SET removed_at=?, removed_reason=? WHERE mention_id=? "
            "AND removed_at IS NULL", (now, f"拆分至 {new_id}", mid))
        conn.execute(
            "INSERT OR IGNORE INTO cluster_membership(membership_id, mention_id, cluster_id, "
            "decision_id, added_at) VALUES (?,?,?,?,?)",
            (det_uuid("mem", mid, new_id, "split"), mid, new_id, decision_id, now))
        # 实体参与边迁移
        conn.execute(
            "INSERT OR IGNORE INTO semantic_relation(relation_id, from_type, from_id, to_type, "
            "to_id, relation, asserted_by, note, created_by, created_at) "
            "SELECT ?, from_type, from_id, to_type, ?, relation, 'manual', '拆簇迁移', 'manual', ? "
            "FROM semantic_relation WHERE relation='participates_in' AND to_id=? AND deleted_at IS NULL",
            (det_uuid("rel-split", mid, new_id), new_id, now, cluster_id))
    conn.execute("UPDATE event_cluster SET state='split', updated_at=? WHERE cluster_id=?",
                 (now, cluster_id))
    conn.commit()
    _refresh_cluster(conn, cluster_id)
    _refresh_cluster(conn, new_id)
    if r["process_id"]:
        _update_precedes(conn, r["process_id"])
    conn.commit()
    observe.emit(conn, "crud", f"拆分簇: {cluster_id[:10]} → {new_id[:10]}（{len(move_set)} 成员）",
                 level="warn", kind="crud.op.split_cluster", target_id=new_id,
                 data={"from": cluster_id, "to": new_id, "members": sorted(move_set),
                       "reason": reason})
    return {"split_from": cluster_id, "new_cluster": new_id, "moved": sorted(move_set)}


def merge_clusters(conn, source_ids: list[str], target_id: str, *, reason: str,
                   expected_version: int) -> dict:
    """高影响操作：仅人工触发（04 §8.2）。源簇 redirect 到目标簇。"""
    target = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                          (target_id,)).fetchone()
    if target is None:
        raise NotFound("目标事件不存在")
    if expected_version != target["version"]:
        raise Conflict(f"目标簇版本冲突: 期望 {expected_version} 实际 {target['version']}")
    if target_id in source_ids:
        raise BadRequest("目标簇不能同时是源簇")
    now = now_iso()
    decision_id = det_uuid("dec-merge", target_id, canonical(sorted(source_ids)), now)
    moved = []
    for sid in source_ids:
        src = conn.execute("SELECT * FROM event_cluster WHERE cluster_id=? AND deleted_at IS NULL",
                           (sid,)).fetchone()
        if src is None:
            raise NotFound(f"源簇不存在: {sid}")
        rows = conn.execute(
            "SELECT mention_id FROM cluster_membership WHERE cluster_id=? AND removed_at IS NULL",
            (sid,)).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE cluster_membership SET removed_at=?, removed_reason=? WHERE mention_id=? "
                "AND removed_at IS NULL", (now, f"合簇至 {target_id}", row["mention_id"]))
            conn.execute(
                "INSERT OR IGNORE INTO cluster_membership(membership_id, mention_id, cluster_id, "
                "decision_id, added_at) VALUES (?,?,?,?,?)",
                (det_uuid("mem", row["mention_id"], target_id, "merge"), row["mention_id"],
                 target_id, decision_id, now))
            moved.append(row["mention_id"])
        conn.execute(
            "UPDATE event_cluster SET state='redirected', redirect_to=?, deleted_at=?, "
            "updated_at=? WHERE cluster_id=?", (target_id, now, now, sid))
        conn.execute(
            "UPDATE semantic_relation SET deleted_at=? WHERE (from_type='event' AND from_id=?) "
            "OR (to_type='event' AND to_id=?)", (now, sid, sid))
    conn.execute(
        "INSERT INTO resolution_decision(decision_id, mention_id, action, target_cluster_id, "
        "reason_code, policy_version, created_at) VALUES (?,?,?,?,?,?,?)",
        (decision_id, f"merge:{target_id}", "manual", target_id, f"人工合簇: {reason}",
         "manual-v1", now))
    conn.commit()
    refreshed = _refresh_cluster(conn, target_id)
    if refreshed["process_id"]:
        _update_precedes(conn, refreshed["process_id"])
    conn.commit()
    observe.emit(conn, "crud", f"合并簇: {len(source_ids)} 个源 → {target_id[:10]}"
                 f"（{len(moved)} 成员迁移）", level="warn", kind="crud.op.merge_clusters",
                 target_id=target_id, data={"sources": source_ids, "moved": len(moved),
                                            "reason": reason})
    return {"merged_into": target_id, "sources": source_ids, "moved": moved,
            "target_version": refreshed["version"]}


def retract_assertion(conn, assertion_id: str, *, basis: str) -> dict:
    """撤回断言：不物理删除，选择历史修订 + 纠正变化卡（04 §9.4 retraction）。"""
    a = conn.execute("SELECT * FROM assertion WHERE assertion_id=?", (assertion_id,)).fetchone()
    if a is None:
        raise NotFound("断言不存在")
    now = now_iso()
    decision_id = det_uuid("dec-retract", assertion_id)
    conn.execute("UPDATE assertion SET status='retracted' WHERE assertion_id=?", (assertion_id,))
    conn.execute(
        "UPDATE slot_selection_history SET sys_to=? WHERE chosen_assertion_id=? AND sys_to IS NULL",
        (now, assertion_id))
    conn.execute(
        "INSERT INTO resolution_decision(decision_id, mention_id, action, reason_code, "
        "policy_version, created_at) VALUES (?,?,?,?,?,?)",
        (decision_id, f"assertion:{assertion_id}", "reject", f"撤回断言: {basis}",
         "manual-v1", now))
    conn.execute(
        "INSERT OR IGNORE INTO change_record(change_id, slot_key, event_id, subject_entity_id, "
        "change_kind, before_json, after_json, importance, dedupe_key, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (det_uuid("chg-retract", assertion_id), a["slot_key"], a["event_id"],
         a["subject_entity_id"], "retraction", a["value_json"], None, "high",
         det_uuid("chg", a["slot_key"], "retraction", assertion_id), now))
    conn.commit()
    observe.emit(conn, "crud", f"撤回断言: {assertion_id[:12]}（{basis}）", level="warn",
                 kind="crud.op.retract_assertion", target_id=assertion_id)
    return {"retracted": assertion_id}
