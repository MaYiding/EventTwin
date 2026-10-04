#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料整理器（DATA_DESIGN.md L1/L2/L3 层落地）。

把 data/corpus/v2/{company}.json 原始批清洗成按【公司 × 月份】分块的 clean 层：
  data/corpus/clean/{company}/{YYYY-MM}.json   （条目按时间升序，带幂等键）
  data/corpus/clean/manifest.json              （文件指纹与统计）
  data/state/corpus_ledger.db                  （corpus_item 全局账本，抽取幂等键）

清洗规则 clean-v1：
  1. norm_url：去 query tracking、去 fragment、小写 host、去尾斜杠；
  2. 正文：去首尾空白 → 句边界截断 ≤800 字；<100 字丢弃（计数）；
  3. published_at 统一 ISO（失败置 null，计数）；
  4. 幂等键：content_hash=sha256(正文)[:16]、url_hash=sha256(norm_url)[:16]；
  5. 去重：公司内同 content_hash 留最早；跨公司同 content_hash 保留但登记 duplicate_of。

幂等：重复运行零变化（输出确定性；账本 INSERT OR IGNORE）。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "corpus" / "v2"
DST = ROOT / "data" / "corpus" / "clean"
LEDGER = ROOT / "data" / "state" / "corpus_ledger.db"
CLEAN_VERSION = "clean-v1"

def norm_url(u: str) -> str:
    """统一实现见 intel/urls.py；此处延迟导入（脚本独立运行）。"""
    sys.path.insert(0, str(ROOT))
    from intel.urls import norm_url as _nu
    return _nu(u)


def h16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def truncate_at_sentence(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", "；", ".", "!", "?", "\n"):
        p = cut.rfind(sep)
        if p > limit // 2:
            return cut[: p + 1]
    return cut


def norm_time(t):
    """尽力转 ISO+08:00；失败返回 None。"""
    if not t:
        return None
    s = str(t).strip().replace("Z", "+00:00")
    fmts = ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y年%m月%d日"]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            from intel.util import CST
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CST)
            return dt.isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def clean_item(it: dict) -> tuple[dict | None, str]:
    """返回 (清洗后条目 | None, 处置原因)。"""
    url = norm_url(it.get("url", ""))
    content = (it.get("content") or "").strip()
    if not url or len(content) < 100:
        return None, "dropped_short"
    content = truncate_at_sentence(re.sub(r"\s+\n", "\n", content))
    pub = norm_time(it.get("published_at"))
    out = {
        "url": url,
        "url_hash": h16(url),
        "content_hash": h16(content),
        "title": (it.get("title") or "").strip()[:200],
        "published_at": pub,
        "fetched_at": it.get("fetched_at"),
        "source_name": it.get("source_name"),
        "language": it.get("language", "zh"),
        "content": content,
        "duplicate_of": None,
    }
    return out, "" if pub else "no_time"


def init_ledger(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS corpus_item (
      content_hash TEXT NOT NULL,
      url_hash     TEXT NOT NULL,
      company      TEXT NOT NULL,
      month        TEXT,
      url          TEXT, title TEXT, published_at TEXT, source_name TEXT,
      content_len  INTEGER NOT NULL,
      first_seen_in TEXT NOT NULL,
      duplicate_of TEXT,
      extracted    INTEGER NOT NULL DEFAULT 0,
      document_version_id TEXT,
      created_at   TEXT NOT NULL,
      PRIMARY KEY (company, content_hash)
    );
    CREATE INDEX IF NOT EXISTS ledger_by_hash ON corpus_item(content_hash);
    CREATE INDEX IF NOT EXISTS ledger_by_company ON corpus_item(company, month);
    CREATE TABLE IF NOT EXISTS extraction_result (
      content_hash TEXT PRIMARY KEY,      -- 全局幂等键：同内容只抽取一次
      mentions     TEXT NOT NULL,         -- 抽取产物 JSON（mention 原始结构）
      standalone   TEXT NOT NULL DEFAULT '[]',
      model_id     TEXT NOT NULL,
      prompt_hash  TEXT NOT NULL,
      usage_count  INTEGER NOT NULL DEFAULT 1,  -- 被复用次数（转载计数）
      created_at   TEXT NOT NULL
    );
    """)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from intel.util import now_iso
    DST.mkdir(parents=True, exist_ok=True)
    ledger = sqlite3.connect(str(LEDGER))
    init_ledger(ledger)

    global_by_hash: dict[str, tuple[str, str]] = {}  # content_hash -> (company, first_seen)
    files_meta, stats = [], Counter()
    month_matrix: dict[str, Counter] = defaultdict(Counter)

    for src in sorted(SRC.glob("*.json")):
        company = src.stem
        batch = json.loads(src.read_text(encoding="utf-8"))
        by_month: dict[str, list[dict]] = defaultdict(list)
        seen_in_company: set[str] = set()
        for it in batch.get("items", []):
            clean, reason = clean_item(it)
            if clean is None:
                stats[reason] += 1
                continue
            if reason:
                stats[reason] += 1
            ch = clean["content_hash"]
            # 公司内去重：同内容只留最早
            if ch in seen_in_company:
                stats["dup_in_company"] += 1
                continue
            seen_in_company.add(ch)
            # 跨公司同内容：保留但登记转载链
            if ch in global_by_hash:
                clean["duplicate_of"] = ch
                stats["dup_cross_company"] += 1
            month = (clean["published_at"] or "unknown")[:7]
            if not re.match(r"^\d{4}-\d{2}$", month):
                month = "unknown"
            by_month[month].append(clean)
            global_by_hash[ch] = (company, f"{company}/{month}.json")
            month_matrix[company][month] += 1
            ledger.execute(
                "INSERT OR IGNORE INTO corpus_item(content_hash, url_hash, company, month, "
                "url, title, published_at, source_name, content_len, first_seen_in, "
                "duplicate_of, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ch, clean["url_hash"], company, month, clean["url"], clean["title"],
                 clean["published_at"], clean["source_name"], len(clean["content"]),
                 f"{company}/{month}.json", clean["duplicate_of"], now_iso()))
            stats["kept"] += 1
        # 写分块文件（月内按时间升序，确定性）
        comp_dir = DST / company
        comp_dir.mkdir(parents=True, exist_ok=True)
        for month in sorted(by_month):
            items = sorted(by_month[month], key=lambda x: (x["published_at"] or "", x["url"]))
            payload = {"company": company, "month": month, "clean_version": CLEAN_VERSION,
                       "items": items}
            fp = comp_dir / f"{month}.json"
            fp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            files_meta.append({
                "path": f"{company}/{month}.json",
                "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()[:16],
                "company": company, "month": month, "n_items": len(items)})
            stats["files"] += 1

    ledger.commit()
    manifest = {
        "clean_version": CLEAN_VERSION,
        "generated_at": now_iso(),
        "source": "data/corpus/v2",
        "files": files_meta,
        "stats": dict(stats),
        "companies": {c: sum(month_matrix[c].values()) for c in sorted(month_matrix)},
    }
    (DST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    ledger.close()
    print(f"整理完成: {dict(stats)}")
    print(f"公司数: {len(month_matrix)}，月文件: {stats['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
