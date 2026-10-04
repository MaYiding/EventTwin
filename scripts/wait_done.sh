#!/bin/bash
# 等待看门狗宣布全部完成，然后输出终态统计。
cd "/Users/mayiding/Desktop/Git/Research/数据库-图/Code"
while ! grep -q "ALL DONE" data/state/watchdog.log 2>/dev/null; do
  sleep 300
done
echo "PIPELINE_ALL_DONE at $(date '+%F %T')"
sqlite3 data/state/intel.db "SELECT status, COUNT(*) FROM document_version GROUP BY status;"
sqlite3 data/state/intel.db "SELECT action, COUNT(*) FROM resolution_decision GROUP BY action;"
sqlite3 data/state/intel.db "SELECT COUNT(*) FROM event_cluster; SELECT COUNT(*) FROM assertion; SELECT COUNT(*) FROM semantic_relation WHERE deleted_at IS NULL;"
