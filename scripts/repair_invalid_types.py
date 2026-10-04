#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复因旧校验器（类型不在 7 类枚举）被误废的提及。

- 只处理 invalid_reason LIKE 'bad_event_type:%' 的行；
- 应用 event_types.normalize_event_type 重映射，保留原始词到 event_type_raw；
- 重跑引文校验（quote 在规范正文中可定位才回活）；
- no_action / no_evidence / quote_not_found 的行不动（那是证据问题，不是类型问题）。

幂等：重复运行零变化。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intel.event_types import normalize_event_type  # noqa: E402
from intel.util import find_quote_span  # noqa: E402


def main() -> int:
    conn = sqlite3.connect(str(ROOT / "data" / "state" / "intel.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT m.mention_id, m.event_type, m.evidence_json, dv.normalized_text "
        "FROM event_mention m "
        "LEFT JOIN document_version dv ON dv.document_version_id=m.document_version_id "
        "WHERE m.status='invalid' AND m.invalid_reason LIKE 'bad_event_type:%' "
        "ORDER BY m.mention_id").fetchall()
    revived, still_bad = 0, 0
    for r in rows:
        mapped, raw = normalize_event_type(r["event_type"])
        # 重验引文
        ok = False
        try:
            for ev in json.loads(r["evidence_json"] or "[]")[:1]:
                if find_quote_span(r["normalized_text"] or "", ev.get("quote", "")) is not None:
                    ok = True
                    break
        except json.JSONDecodeError:
            ok = False
        if ok:
            conn.execute(
                "UPDATE event_mention SET event_type=?, event_type_raw=?, status='valid', "
                "invalid_reason=NULL WHERE mention_id=?",
                (mapped, raw, r["mention_id"]))
            revived += 1
        else:
            # 类型修复但引文确实定位不到 → 保持 invalid，原因改为真实原因
            conn.execute(
                "UPDATE event_mention SET event_type=?, event_type_raw=?, "
                "invalid_reason='quote_not_found_after_type_fix' WHERE mention_id=?",
                (mapped, raw, r["mention_id"]))
            still_bad += 1
    conn.commit()
    total_valid = conn.execute(
        "SELECT COUNT(*) FROM event_mention WHERE status='valid'").fetchone()[0]
    print(f"处理 {len(rows)} 条误废提及：回活 {revived}，仍无效 {still_bad}（引文定位失败）")
    print(f"当前有效提及总数: {total_valid}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
