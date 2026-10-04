#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语法与导入自检：编译全部模块 + 关键导入 + schema 初始化到内存库。"""
from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAIL = 0


def check(msg: str, fn):
    global FAIL
    try:
        fn()
        print(f"  ✓ {msg}")
    except Exception as e:  # noqa: BLE001
        FAIL += 1
        print(f"  ✗ {msg}: {e}")


print("== 编译全部 Python 模块 ==")
for p in sorted(ROOT.glob("intel/**/*.py")) + sorted(ROOT.glob("scripts/*.py")):
    check(p.relative_to(ROOT), lambda p=p: py_compile.compile(str(p), doraise=True))

print("== 关键导入 ==")
check("import intel.llm", lambda: __import__("intel.llm", fromlist=["llm"]))
check("import intel.pipeline.runner", lambda: __import__("intel.pipeline.runner", fromlist=["x"]))
check("import intel.graph_service", lambda: __import__("intel.graph_service", fromlist=["x"]))
check("import intel.query_service", lambda: __import__("intel.query_service", fromlist=["x"]))
check("import intel.server", lambda: __import__("intel.server", fromlist=["x"]))

print("== 内存库 schema 初始化 ==")
import sqlite3  # noqa: E402
from intel.store import db  # noqa: E402

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = sqlite3.connect(tmp.name)
conn.row_factory = sqlite3.Row
db.init_db(conn)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
expect = {"source", "raw_snapshot", "document", "document_version", "source_lineage",
          "entity", "entity_alias", "event_mention", "event_cluster", "cluster_membership",
          "cluster_version", "process", "assertion", "assertion_evidence",
          "slot_selection_history", "semantic_relation", "resolution_decision",
          "pipeline_run", "outbox", "projection_state", "queue", "pipeline_event",
          "change_record", "subscription", "delivery", "vector"}
missing = expect - set(tables)
check(f"26 张核心表齐全（缺: {missing or '无'}）", lambda: (_ for _ in ()).throw(AssertionError(missing))
      if missing else None)
print("== 完成 ==")
sys.exit(1 if FAIL else 0)
