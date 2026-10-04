# -*- coding: utf-8 -*-
"""采集与入库（Raw/Truth Ledger 第一段）：
语料文件 → raw_snapshot(内容寻址) → document/document_version → source_lineage → 队列。

两层去重对齐 04 §4.2：
1. 字节/正文重复：内容哈希相同 → 新文档版本标记 duplicate_of，复用解析抽取结果；
2. 近重复/转载：先按 lineage_group（内容指纹哈希）归组，转载不增加独立来源数。
"""
from __future__ import annotations

import json
import re
import sqlite3
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .. import config as cfg
from .. import observe
from ..store import objects, queue
from ..util import det_uuid, iso, now_iso, parse_datetime, sha256_text


def canonical_url(url: str) -> str:
    """URL 规范化（唯一实现 intel/urls.py，与语料整理器 clean 层同规则）。"""
    from ..urls import norm_url
    return norm_url(url)


def _source_id(source_name: str) -> str:
    return det_uuid("source", source_name.strip() or "unknown")


def ensure_source(conn: sqlite3.Connection, name: str) -> str:
    sid = _source_id(name)
    conn.execute(
        "INSERT OR IGNORE INTO source(source_id, name, adapter, created_at) VALUES (?,?,?,?)",
        (sid, name.strip() or "unknown", "corpus", now_iso()),
    )
    return sid


def ingest_item(conn: sqlite3.Connection, item: dict) -> str | None:
    """入库单条资料；返回 document_version_id（重复内容返回新版本行，标记 duplicate_of）。"""
    url = item.get("url") or item.get("origin_id") or ""
    curl = canonical_url(url) if url else ""
    source_name = (item.get("source_name") or (urlsplit(curl).netloc if curl else "未知来源"))
    sid = ensure_source(conn, source_name)
    origin_id = curl or sha256_text(item.get("title") or "")
    document_id = det_uuid("doc", sid, origin_id)

    title = (item.get("title") or "").strip()
    content = (item.get("content") or item.get("text") or "").strip()
    if not content or len(content) < 60:
        observe.emit(conn, "ingest", f"跳过内容过短资料: {title[:40]}", level="warn",
                     kind="ingest.skip", data={"url": url})
        return None
    published_dt, _prec = parse_datetime(item.get("published_at"))
    fetched_dt, _ = parse_datetime(item.get("fetched_at"))
    fetched_iso = iso(fetched_dt) if fetched_dt else now_iso()

    payload = json.dumps({
        "url": url, "title": title, "source_name": source_name,
        "published_at": item.get("published_at"), "language": item.get("language", "zh"),
        "content": content,
    }, ensure_ascii=False).encode("utf-8")
    content_hash, _path = objects.put(payload)

    # 字节级重复检测：同内容已存在 → 本文档只登记来源关联（duplicate_of）
    dup = conn.execute(
        "SELECT dv.document_version_id FROM document_version dv "
        "JOIN raw_snapshot rs ON rs.snapshot_id=dv.snapshot_id "
        "WHERE rs.content_hash=? AND dv.document_id<>? LIMIT 1",
        (content_hash, document_id),
    ).fetchone()

    # 版本号推进：同 URL 内容变化 → 新版本（模拟官网改价形成新 DocumentVersion）
    prev = conn.execute(
        "SELECT version_no, snapshot_id FROM document_version WHERE document_id=? "
        "ORDER BY version_no DESC LIMIT 1", (document_id,)).fetchone()
    version_no = (prev["version_no"] + 1) if prev else 1
    # 幂等判定：该内容在本文档任意历史版本出现过即跳过。
    # 只比"最新版本"会让同 URL 不同内容的两条目在每次重跑 ingest 时交替新建版本
    # （重跑 N 次 → 多 2N 个版本），破坏续跑与重放一致性。
    seen_any = conn.execute(
        "SELECT 1 FROM document_version dv JOIN raw_snapshot rs ON rs.snapshot_id=dv.snapshot_id "
        "WHERE dv.document_id=? AND rs.content_hash=? LIMIT 1",
        (document_id, content_hash),
    ).fetchone()
    if seen_any:
        return None  # 同文档同内容，重复投递安全（幂等）

    docv_id = det_uuid("docv", document_id, str(version_no))
    snapshot_id = det_uuid("snap", content_hash)
    conn.execute(
        "INSERT OR IGNORE INTO raw_snapshot(snapshot_id, content_hash, uri, fetched_at, bytes, meta_json) "
        "VALUES (?,?,?,?,?,?)",
        (snapshot_id, content_hash, url, fetched_iso, len(payload),
         json.dumps({"title": title}, ensure_ascii=False)),
    )
    status = "captured"
    note = None
    if dup:
        status = "duplicate_of"
        note = dup["document_version_id"]
    conn.execute(
        "INSERT INTO document(document_id, source_id, origin_id, canonical_url, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(document_id) DO NOTHING",
        (document_id, sid, origin_id, curl, now_iso()),
    )
    # 解析（parse 阶段）：正文即抓取文本，规范化空白后存 normalized_text
    normalized = _normalize_text(content)
    conn.execute(
        "INSERT INTO document_version(document_version_id, document_id, version_no, snapshot_id, "
        "title, published_at, fetched_at, parse_version, normalized_text, status, dedupe_of, note, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (docv_id, document_id, version_no, snapshot_id, title,
         iso(published_dt) if published_dt else None, fetched_iso,
         "text-v1", normalized, "parsed" if not dup else status, note, note, now_iso()),
    )
    # 来源谱系：lineage_group = 内容指纹（近重复组）；字节重复沿用被复用文档的组
    if dup:
        lg = conn.execute("SELECT lineage_group FROM source_lineage WHERE document_version_id=?",
                          (dup["document_version_id"],)).fetchone()
        lineage_group = lg["lineage_group"] if lg else content_hash[:16]
        kind = "reprint"
    else:
        lineage_group = _near_dupe_group(conn, normalized)
        kind = "original"
    conn.execute(
        "INSERT OR REPLACE INTO source_lineage(document_version_id, lineage_group, kind, "
        "based_on_hash, created_at) VALUES (?,?,?,?,?)",
        (docv_id, lineage_group, kind, content_hash if dup else None, now_iso()),
    )
    if not dup:
        queue.enqueue(conn, "intel.document.parsed.v1",
                      {"document_version_id": docv_id, "idempotency_key": f"extract:{docv_id}"},
                      trace_id=docv_id)
    observe.emit(conn, "ingest", f"入库: {title[:50]}",
                 kind="ingest.stored", target_id=docv_id,
                 data={"source": source_name, "url": url, "chars": len(content),
                       "duplicate": bool(dup), "version_no": version_no})
    return docv_id


def _normalize_text(text: str) -> str:
    """规范化正文：去多余空白行，保留数字/否定词/单位（去重规范化不能吞掉事实差异）。"""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _near_dupe_group(conn: sqlite3.Connection, normalized: str) -> str:
    """近重复分组：数字与实体保留的前提下，取前 2000 字符指纹哈希作组键。"""
    return sha256_text(normalized[:2000])[:16]


def run(conn: sqlite3.Connection, corpus_dir=None) -> dict:
    """扫描语料目录（稳定排序保证可复现），逐条入库。

    默认扫 config.pipeline.corpus_dirs 列出的全部目录（v1 场景批 + v2 企业长历史批）。
    """
    cfg.ensure_dirs()
    if corpus_dir is not None:
        dirs = [corpus_dir]
    else:
        from pathlib import Path as _P
        dirs = [_P(cfg.ROOT) / d for d in
                cfg.load_config().get("pipeline", {}).get("corpus_dirs", ["data/corpus/incoming"])]
    files = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        files.extend(sorted(p for p in d.glob("*.json") if p.is_file()))
    files.sort(key=lambda p: (p.name, str(p)))
    stats = {"files": 0, "items": 0, "stored": 0, "duplicates": 0, "skipped": 0}
    for fp in files:
        try:
            batch = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            observe.emit(conn, "ingest", f"语料文件损坏: {fp.name} {e}", level="error")
            continue
        stats["files"] += 1
        items = batch.get("items") if isinstance(batch, dict) else batch
        # 稳定排序：同 URL 去重、按发布时间有序入库
        items = sorted([i for i in (items or []) if isinstance(i, dict)],
                       key=lambda x: (str(x.get("published_at") or ""), str(x.get("url") or "")))
        for item in items:
            stats["items"] += 1
            with conn:
                docv = ingest_item(conn, item)
            if docv:
                stats["stored"] += 1
                row = conn.execute("SELECT status FROM document_version WHERE document_version_id=?",
                                   (docv,)).fetchone()
                if row and row["status"] == "duplicate_of":
                    stats["duplicates"] += 1
            else:
                stats["skipped"] += 1
    observe.emit(conn, "ingest", f"语料入库完成: {stats}", kind="ingest.done", data=stats)
    return stats
