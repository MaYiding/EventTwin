# -*- coding: utf-8 -*-
"""事实更新与双时间账本（04 §9）——Assertion 层。

九种决定：duplicate / support / enrich / state_transition / conflict / correction /
retraction / expire / needs_review。
铁律：
- 断言只追加、不覆盖；当前选择另存 slot_selection_history（valid_during × sys_during 双时间）；
- 不用"最新报道胜出"：同口径重叠期不相容值 → conflict 保留争议；
- 迟到更正 = 关闭旧选择的系统区间，按真实有效时间拆分追加新行（不是物理删除）；
- 变化卡只对白名单谓词与真实语义变化生成，转载不触发。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .. import config as cfg, observe
from ..util import det_uuid, now_iso, parse_datetime, iso, CST
from . import entities as entities_mod
from .resolve import _add_edge

PRED_LABELS = {"list_price": "标价", "price": "价格", "release_date": "发布日期",
               "status": "状态", "amount": "金额", "equity_share": "股权比例"}


# ---------------------------------------------------------------------------
# 槽位与断言构造
# ---------------------------------------------------------------------------

def scope_key(scope: dict) -> str:
    parts = [f"{k}={v}" for k, v in sorted((scope or {}).items())
             if k != "business_no" and v not in (None, "", [])]
    return "|".join(parts)


def make_slot_key(subject_entity_id: str, predicate: str, scope: dict) -> str:
    return f"{subject_entity_id}:{predicate}:{scope_key(scope)}"


def _value_json(claim: dict) -> dict:
    return {"value": str(claim.get("value", "")), "unit": claim.get("unit"),
            "currency": claim.get("currency"), "tax_included": claim.get("tax_included"),
            "metric": claim.get("metric")}


def _values_equal(a: dict, b: dict) -> bool:
    va, vb = str(a.get("value", "")).strip(), str(b.get("value", "")).strip()
    if _num(va) is not None and _num(vb) is not None:
        return abs(_num(va) - _num(vb)) < 1e-9
    return va == vb


def _num(s: str):
    try:
        return float(str(s).replace(",", "").replace("元", "").replace("%", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 提及 claims → 断言更新
# ---------------------------------------------------------------------------

def process_mention_claims(conn: sqlite3.Connection, payload: dict) -> dict:
    """处理 resolve 阶段入队的 claims 任务（mention → 断言 → 选择 → 变化卡）。"""
    mention_id = payload.get("mention_id")
    mention = conn.execute("SELECT * FROM event_mention WHERE mention_id=?", (mention_id,)).fetchone()
    if mention is None:
        return {"skipped": True}
    claims = json.loads(mention["claims_json"] or "[]")
    if not claims:
        return {"no_claims": True}
    doc = conn.execute(
        "SELECT dv.document_version_id, d.canonical_url, sl.lineage_group, s.name AS source_name "
        "FROM document_version dv JOIN document d ON d.document_id=dv.document_id "
        "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
        "LEFT JOIN source s ON s.source_id=d.source_id "
        "WHERE dv.document_version_id=?", (mention["document_version_id"],)).fetchone()
    actors = [a for a in json.loads(mention["actor_json"] or "[]") if isinstance(a, dict)]
    objects = [o for o in json.loads(mention["object_json"] or "[]") if isinstance(o, dict)]
    m_scope = json.loads(mention["scope_json"] or "{}")
    stats = {}
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("value") in (None, ""):
            continue
        predicate = claim.get("predicate") or "other"
        # 谓词主体：产品属性（价格/日期）挂对象实体，交易类挂参与方
        if predicate in ("list_price", "price", "release_date") and objects:
            subject = objects[0]
        elif actors:
            subject = actors[0]
        elif objects:
            subject = objects[0]
        else:
            continue
        sid = subject.get("entity_id")
        if not sid:
            sid = entities_mod.resolve_entity(conn, subject.get("name", ""), subject.get("type"))
        if not sid:
            continue
        scope = dict(m_scope)
        for k in ("market", "channel", "sku"):
            if claim.get(k):
                scope[k] = claim[k]
        slot_key = make_slot_key(sid, predicate, scope)
        vf, vt = _valid_during(claim, mention)
        decision = upsert_assertion(
            conn, slot_key=slot_key, subject_id=sid, predicate=predicate, scope=scope,
            value=_value_json(claim), valid_from=vf, valid_to=vt,
            docver=doc, mention=mention, event_id=payload.get("event_id"),
            correction_hint=bool(mention["correction_hint"]) or bool(claim.get("correction_hint")),
            evidence=_mention_evidence(mention))
        stats[predicate] = decision["decision"]
    return stats or {"no_effective_claims": True}


def process_standalone(conn: sqlite3.Connection, payload: dict) -> dict:
    """无事件动作的状态断言（页面只列"现价 1899"这类），event_id=null。"""
    sa = payload.get("assertion") or {}
    quote = payload.get("quote") or ""
    doc = conn.execute(
        "SELECT dv.document_version_id, d.canonical_url, sl.lineage_group, s.name AS source_name "
        "FROM document_version dv JOIN document d ON d.document_id=dv.document_id "
        "LEFT JOIN source_lineage sl ON sl.document_version_id=dv.document_version_id "
        "LEFT JOIN source s ON s.source_id=d.source_id "
        "WHERE dv.document_version_id=?",
        (payload.get("document_version_id"),)).fetchone()
    subject_name = sa.get("subject") or ""
    sid = entities_mod.resolve_entity(conn, subject_name, None)
    if not sid:
        return {"skipped": "no_subject"}
    predicate = sa.get("predicate") or "other"
    scope = {k: v for k, v in (sa.get("scope") or {}).items() if v}
    vf_dt, _ = parse_datetime(sa.get("valid_from"))
    vf_iso = iso(vf_dt) if vf_dt else None
    vt_dt, _ = parse_datetime(sa.get("valid_to"))
    vt_iso = iso(vt_dt) if vt_dt else None
    return upsert_assertion(
        conn, slot_key=make_slot_key(sid, predicate, scope), subject_id=sid,
        predicate=predicate, scope=scope,
        value={"value": str(sa.get("value", "")), "unit": sa.get("unit")},
        valid_from=vf_iso, valid_to=vt_iso, docver=doc, mention=None, event_id=None,
        correction_hint=bool(sa.get("correction_hint")),
        evidence=[{"quote": quote}] if quote else [])


# ---------------------------------------------------------------------------
# 核心：断言追加 + 选择更新 + 变化卡
# ---------------------------------------------------------------------------

def upsert_assertion(conn, *, slot_key, subject_id, predicate, scope, value, valid_from,
                     valid_to, docver, mention, event_id, correction_hint, evidence) -> dict:
    now = now_iso()
    docver = dict(docver) if docver is not None else {}  # sqlite3.Row → dict
    assertion_id = det_uuid("asrt", slot_key, canonical(value), valid_from or "",
                            (docver or {}).get("document_version_id", ""))
    lineage = (docver or {}).get("lineage_group")
    conn.execute(
        "INSERT OR IGNORE INTO assertion(assertion_id, slot_key, subject_entity_id, predicate, "
        "scope_json, value_json, valid_from, valid_to, time_quality, document_version_id, "
        "mention_id, event_id, lineage_group, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (assertion_id, slot_key, subject_id, predicate,
         json.dumps(scope, ensure_ascii=False), json.dumps(value, ensure_ascii=False),
         valid_from, valid_to, "explicit" if valid_from else "unknown",
         (docver or {}).get("document_version_id"), mention["mention_id"] if mention else None,
         event_id, lineage, now))
    for i, ev in enumerate(evidence[:3]):
        conn.execute(
            "INSERT OR IGNORE INTO assertion_evidence(assertion_id, evidence_no, "
            "document_version_id, quote, url, relation) VALUES (?,?,?,?,?,?)",
            (assertion_id, i, (docver or {}).get("document_version_id"),
             ev.get("quote"), (docver or {}).get("canonical_url"), "supports"))

    # 已有同槽位活跃断言（按创建时间）
    priors = conn.execute(
        "SELECT * FROM assertion WHERE slot_key=? AND status='active' "
        "AND assertion_id<>? ORDER BY assertion_id", (slot_key, assertion_id)).fetchall()

    decision = _classify(conn, value, valid_from, valid_to, correction_hint, lineage, priors)
    _apply_decision(conn, decision, slot_key, assertion_id, value, valid_from, valid_to,
                    subject_id, predicate, event_id, now)
    observe.emit(conn, "assertion",
                 f"断言[{decision['decision']}]: {slot_key.split(':')[-1] and PRED_LABELS.get(predicate, predicate)}"
                 f" = {value.get('value')}",
                 kind="assertion.decision", target_id=assertion_id,
                 data={"slot_key": slot_key, "decision": decision["decision"],
                       "value": value, "valid_from": valid_from, "prior_count": len(priors)})
    return decision


def _classify(conn, value, valid_from, valid_to, correction_hint, lineage, priors) -> dict:
    if not priors:
        return {"decision": "first_seen", "prior": None}
    latest = priors[-1]
    same_lineage_same_value = (latest["lineage_group"] == lineage) and \
        _values_equal(json.loads(latest["value_json"]), value)
    if same_lineage_same_value:
        return {"decision": "duplicate", "prior": latest}
    if _values_equal(json.loads(latest["value_json"]), value):
        return {"decision": "support", "prior": latest}
    if correction_hint:
        return {"decision": "correction", "prior": latest}
    lf, nf = _dt(latest["valid_from"]), _dt(valid_from)
    if lf is None or nf is None:
        return {"decision": "needs_review", "prior": latest}
    if nf > lf:
        # 同槽位较晚有效期的新变化（且旧值没有与之重叠冲突的更正说明）
        return {"decision": "state_transition", "prior": latest}
    if nf == lf:
        lt, nt = _dt(latest["valid_to"]), _dt(valid_to)
        if lt is not None and nt is not None and lt >= nt:
            return {"decision": "needs_review", "prior": latest}
        return {"decision": "conflict", "prior": latest}
    return {"decision": "needs_review", "prior": latest}  # 迟到更旧的事实


def _apply_decision(conn, dec, slot_key, assertion_id, value, valid_from, valid_to,
                    subject_id, predicate, event_id, now) -> None:
    d = dec["decision"]
    prior = dec["prior"]
    decision_id = det_uuid("seldec", assertion_id)
    if d in ("duplicate",):
        return  # 只加来源关联，不加选择、不触发变化
    if d in ("first_seen", "support", "enrich"):
        if d == "first_seen":
            _close_open_selections(conn, slot_key, now, decision_id)
            _insert_selection(conn, slot_key, assertion_id, "accepted", valid_from, valid_to,
                              [assertion_id], decision_id, now)
            _change(conn, slot_key, event_id, subject_id, "first_seen", None, value, now)
        return
    if d == "state_transition":
        old_val = json.loads(prior["value_json"])
        # 旧选择关闭：有效期截到新值生效点，系统区间关闭
        _close_open_selections(conn, slot_key, now, decision_id,
                               new_valid_end=valid_from)
        _insert_selection(conn, slot_key, assertion_id, "accepted", valid_from, valid_to,
                          [assertion_id], decision_id, now)
        _change(conn, slot_key, event_id, subject_id, "state_transition", old_val, value, now)
        return
    if d == "correction":
        old_val = json.loads(prior["value_json"])
        conn.execute("UPDATE assertion SET correction_of=? WHERE assertion_id=?",
                     (prior["assertion_id"], assertion_id))
        _add_edge(conn, "assertion", assertion_id, "assertion", prior["assertion_id"],
                  "corrects", asserted_by="rule", note="官方更正")
        # 纠正历史：关闭旧 sys 区间；按真实有效时间拆分重写选择
        _close_open_selections(conn, slot_key, now, decision_id)
        _insert_selection(conn, slot_key, assertion_id, "accepted", valid_from or prior["valid_from"],
                          valid_to, [assertion_id], decision_id, now)
        _change(conn, slot_key, event_id, subject_id, "correction", old_val, value, now)
        return
    if d == "conflict":
        cand = [prior["assertion_id"], assertion_id]
        _close_open_selections(conn, slot_key, now, decision_id)
        _insert_selection(conn, slot_key, None, "conflicted", valid_from, valid_to, cand,
                          decision_id, now)
        _change(conn, slot_key, event_id, subject_id, "conflict",
                json.loads(prior["value_json"]), value, now)
        _add_edge(conn, "assertion", assertion_id, "assertion", prior["assertion_id"],
                  "contradicts", asserted_by="rule", note="同口径不相容值")
        return
    if d == "needs_review":
        observe.emit(conn, "assertion", f"待复核断言: {slot_key}", level="warn",
                     kind="assertion.needs_review", target_id=assertion_id)
        return


def _close_open_selections(conn, slot_key, now, decision_id, new_valid_end=None) -> None:
    """关闭该槽位所有开放系统区间的选择行。

    new_valid_end 非 None（状态前移）时：旧选择的 valid_to 同时截断到新值生效点，
    体现"9 月 10 日起 1999 的选择有效期结束"；sys 区间关闭到当前时刻。
    """
    rows = conn.execute(
        "SELECT * FROM slot_selection_history WHERE slot_key=? AND sys_to IS NULL",
        (slot_key,)).fetchall()
    for r in rows:
        if new_valid_end and (r["valid_to"] is None or r["valid_to"] > new_valid_end):
            conn.execute(
                "UPDATE slot_selection_history SET sys_to=?, valid_to=? WHERE selection_id=?",
                (now, new_valid_end, r["selection_id"]))
        else:
            conn.execute("UPDATE slot_selection_history SET sys_to=? WHERE selection_id=?",
                         (now, r["selection_id"]))


def _insert_selection(conn, slot_key, chosen_assertion_id, disposition, valid_from, valid_to,
                      cand_ids, decision_id, now) -> None:
    sel_id = det_uuid("sel", slot_key, str(chosen_assertion_id), str(valid_from), disposition)
    conn.execute(
        "INSERT OR IGNORE INTO slot_selection_history(selection_id, slot_key, "
        "chosen_assertion_id, disposition, candidate_ids_json, valid_from, valid_to, "
        "sys_from, sys_to, decision_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (sel_id, slot_key, chosen_assertion_id, disposition,
         json.dumps(cand_ids, ensure_ascii=False), valid_from or _open_bound(),
         valid_to, now, None, decision_id, now))


def _open_bound() -> str:
    return "1970-01-01T00:00:00+08:00"  # 未知有效起点用开区间下界占位（time_quality=unknown 已记录）


def _change(conn, slot_key, event_id, subject_id, kind, before, after, now) -> None:
    watch = cfg.load_config()["changes"]["watch_predicates"]
    predicate = slot_key.split(":")[1] if ":" in slot_key else ""
    if predicate not in watch and kind not in ("conflict",):
        return  # 非白名单谓词不推送（文字润色不算变化）
    dedupe = det_uuid("chg", slot_key, kind, canonical(after or {}), canonical(before or {}))
    try:
        conn.execute(
            "INSERT OR IGNORE INTO change_record(change_id, slot_key, event_id, "
            "subject_entity_id, change_kind, before_json, after_json, importance, "
            "dedupe_key, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (dedupe, slot_key, event_id, subject_id, kind,
             json.dumps(before, ensure_ascii=False) if before else None,
             json.dumps(after, ensure_ascii=False), "normal", dedupe, now))
    except sqlite3.IntegrityError:
        pass  # 幂等：同一变化多个转载不重复


def _mention_evidence(mention) -> list[dict]:
    evs = json.loads(mention["evidence_json"] or "[]")
    return [{"quote": e.get("quote", "")} for e in evs if isinstance(e, dict)]


def _valid_during(claim: dict, mention):
    vf_dt, _ = parse_datetime(claim.get("valid_from"))
    vt_dt, _ = parse_datetime(claim.get("valid_to"))
    if vf_dt is None:
        vf_dt, _ = parse_datetime(mention["event_time_lower"])
    return (iso(vf_dt) if vf_dt else None), (iso(vt_dt) if vt_dt else None)


def _dt(s):
    if not s:
        return None
    dt, _ = parse_datetime(s)
    return dt


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 阶段入口
# ---------------------------------------------------------------------------

def run(conn: sqlite3.Connection) -> dict:
    from ..store import queue as Q
    stats = {"claims_jobs": 0, "standalone": 0, "decisions": {}}
    # 1) 提及 claims 任务
    while True:
        msg = Q.claim(conn, "intel.assertion.candidate.v1")
        if msg is None:
            break
        payload = json.loads(msg["payload_json"])
        if payload.get("mention_id"):
            out = process_mention_claims(conn, payload)
            stats["claims_jobs"] += 1
            if isinstance(out, dict):
                for k, v in out.items():
                    # decisions 键只收字符串决策名：out 的值可能是 bool/数字（如 changed=True），
                    # 混入会让 json.dumps(sort_keys=True) 在 bool<str 比较时崩溃
                    if isinstance(v, str):
                        stats["decisions"][v] = stats["decisions"].get(v, 0) + 1
        else:
            out = process_standalone(conn, payload)
            stats["standalone"] += 1
            dec = out.get("decision") if isinstance(out, dict) else None
            if dec:
                stats["decisions"][dec] = stats["decisions"].get(dec, 0) + 1
        conn.commit()
        Q.complete(conn, msg["message_id"])
    observe.emit(conn, "assertion", f"事实更新阶段完成: {stats}", kind="assertion.stage_done",
                 data=stats)
    return stats
