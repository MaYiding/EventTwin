#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流水线 CLI 入口。

用法：
  python3 scripts/run_pipeline.py                 # 全量跑（自动建库，幂等）
  python3 scripts/run_pipeline.py --reset         # 清空数据库后全量跑（LLM 缓存保留 → 快速重放）
  python3 scripts/run_pipeline.py --reset --fresh # 连 LLM 缓存一起清（真实重算）
  python3 scripts/run_pipeline.py --stages resolve,assertion   # 只跑指定阶段
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intel import config as cfg  # noqa: E402
from intel.observe import emit  # noqa: E402
from intel.pipeline import runner  # noqa: E402
from intel.store import db  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="企业外部情报系统流水线")
    ap.add_argument("--reset", action="store_true", help="清空数据库（保留 LLM 缓存，重放可复现）")
    ap.add_argument("--fresh", action="store_true", help="连同 LLM 缓存一起清空（真实重算）")
    ap.add_argument("--stages", type=str, default=None,
                    help="逗号分隔的阶段子集: ingest,extract,entity,vector,resolve,assertion,project")
    ap.add_argument("--no-cache", action="store_true", help="本次运行绕过 LLM 缓存")
    args = ap.parse_args()

    cfg.ensure_dirs()
    conn = db.connect(cfg.DB_PATH)
    db.init_db(conn)
    if args.reset or args.fresh:
        runner.reset(conn, hard=args.fresh)
        emit(conn, "replay", f"数据库已重置（fresh={args.fresh}）", kind="replay.reset")
        conn.commit()
    stages = args.stages.split(",") if args.stages else None
    results = runner.run_pipeline(conn, stages=stages, use_cache=not args.no_cache)
    print("\n===== 流水线结果 =====")
    for st, out in results.items():
        print(f"  {st}: {out}")
    print("\n===== 全库统计 =====")
    for k, v in runner.stats(conn).items():
        print(f"  {k}: {v}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
