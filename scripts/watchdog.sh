#!/bin/bash
# 流水线看门狗：每 5 分钟检查，进程死且未完成则自动重启续跑（缓存保证幂等）。
# 全部阶段完成后（无 pending 提及且 resolve 已跑过）自动退出。
set -u
cd "/Users/mayiding/Desktop/Git/Research/数据库-图/Code"
PY=/opt/anaconda3/bin/python3
LOG=data/state/watchdog.log
echo "[watchdog] start $(date '+%F %T')" >> "$LOG"
while true; do
  DONE_EXTRACT=$($PY - <<'EOF' 2>/dev/null
import sqlite3
c = sqlite3.connect("data/state/intel.db")
n = c.execute("SELECT COUNT(*) FROM document_version WHERE status IN ('parsed','failed')").fetchone()[0]
print(n)
EOF
)
  ALIVE=$(ps aux | grep "[r]un_pipeline" | wc -l | tr -d ' ')
  TS=$(date '+%F %T')
  if [ "${ALIVE}" = "0" ] && [ "${DONE_EXTRACT}" != "0" ]; then
    echo "[watchdog] $TS pipeline dead, pending=$DONE_EXTRACT → restart" >> "$LOG"
    nohup $PY scripts/run_pipeline.py > "data/state/watchdog_run_$(date '+%H%M').log" 2>&1 &
    sleep 120
    continue
  fi
  # 完成判定：无 parsed/failed 文档、无待归属提及、resolve 阶段有完成记录
  FINISHED=$($PY - <<'EOF' 2>/dev/null
import sqlite3
c = sqlite3.connect("data/state/intel.db")
q1 = c.execute("SELECT COUNT(*) FROM document_version WHERE status IN ('parsed','failed')").fetchone()[0]
q2 = c.execute("SELECT COUNT(*) FROM event_mention m WHERE m.status='valid' AND NOT EXISTS (SELECT 1 FROM cluster_membership cm WHERE cm.mention_id=m.mention_id AND cm.removed_at IS NULL)").fetchone()[0]
q3 = c.execute("SELECT COUNT(*) FROM event_mention WHERE status='valid'").fetchone()[0]
q4 = c.execute("SELECT COUNT(*) FROM pipeline_run WHERE stage='resolve' AND status='completed'").fetchone()[0]
print("YES" if (q1 == 0 and q3 > 0 and q2 == 0 and q4 > 0) else "NO")
EOF
)
  echo "[watchdog] $TS alive=$ALIVE parsed_or_failed=$DONE_EXTRACT finished=$FINISHED" >> "$LOG"
  if [ "$FINISHED" = "YES" ]; then
    echo "[watchdog] $TS ALL DONE" >> "$LOG"
    break
  fi
  sleep 300
done
