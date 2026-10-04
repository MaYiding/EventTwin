#!/usr/bin/env bash
# CRUD 增删改查 + 查询/问答接口冒烟测试（对运行中的 server 执行）
# 用法: bash scripts/crud_smoke.sh [base_url]
set -euo pipefail
BASE="${1:-http://127.0.0.1:8600}"
PASS=0; FAIL=0
say()  { printf "%s\n" "$*"; }
ok()   { PASS=$((PASS+1)); say "  ✓ $1"; }
bad()  { FAIL=$((FAIL+1)); say "  ✗ $1（HTTP $2）: $3"; }
code() { curl -s -o /tmp/smoke_body -w "%{http_code}" "$@"; }
body() { cat /tmp/smoke_body; }

say "== 0. 健康检查 =="
C=$(code "$BASE/api/health"); [ "$C" = 200 ] && ok "health 200" || bad "health" "$C" "$(body)"

say "== 1. 实体 增/查/改/删 =="
RUNTS=$(date +%s)
ENT_NAME="冒烟测试科技-${RUNTS}有限公司"
C=$(code -X POST "$BASE/api/entities" -H 'Content-Type: application/json' \
  -d "{\"name\":\"$ENT_NAME\",\"type\":\"company\",\"aliases\":[\"冒烟测试${RUNTS}\",\"SmokeTest${RUNTS}\"]}"); \
  [ "$C" = 201 ] && ok "POST /api/entities 201" || bad "create entity" "$C" "$(body)"
EID=$(body | ${PY:-python3} -c 'import json,sys; print(json.load(sys.stdin)["entity_id"])')
C=$(code "$BASE/api/entities/$EID"); [ "$C" = 200 ] && ok "GET 实体 200（含别名与事实）" || bad "get entity" "$C" "$(body)"
C=$(code -X PATCH "$BASE/api/entities/$EID" -H 'Content-Type: application/json' \
  -d "{\"name\":\"$ENT_NAME\",\"add_aliases\":[\"冒烟集团${RUNTS}\"],\"expected_updated\":null}")
[ "$C" = 200 ] && ok "PATCH 实体 200（追加别名）" || bad "patch entity" "$C" "$(body)"
C=$(code -X DELETE "$BASE/api/entities/$EID" -H 'Content-Type: application/json' -d '{}')
[ "$C" = 200 ] && ok "DELETE 实体 200（软删除）" || bad "delete entity" "$C" "$(body)"
C=$(code "$BASE/api/entities/$EID"); [ "$C" = 404 ] && ok "删除后 GET 404" || bad "get deleted" "$C" "$(body)"

say "== 2. 人工事件 增/查/改（含 409 乐观锁）/删 =="
C=$(code -X POST "$BASE/api/manual-events" -H 'Content-Type: application/json' \
  -d '{"event_type":"other","title":"冒烟测试事件-发布会彩排","event_time_lower":"2026-09-17T10:00:00+08:00","entity_ids":[]}')
[ "$C" = 201 ] && ok "POST 人工事件 201" || bad "create event" "$C" "$(body)"
CID=$(body | ${PY:-python3} -c 'import json,sys; print(json.load(sys.stdin)["cluster_id"])')
C=$(code "$BASE/api/events/cluster/$CID"); [ "$C" = 200 ] && ok "GET 事件 200（含版本列表）" || bad "get event" "$C" "$(body)"
VER=$(body | ${PY:-python3} -c 'import json,sys; print(json.load(sys.stdin)["version"])')
C=$(code -X PATCH "$BASE/api/manual-events/$CID" -H 'Content-Type: application/json' \
  -d "{\"expected_version\":$VER,\"summary\":\"冒烟测试事件-已改名\"}")
[ "$C" = 200 ] && ok "PATCH 事件 200（v${VER} 升至 v$((VER+1))）" || bad "patch event" "$C" "$(body)"
C=$(code -X PATCH "$BASE/api/manual-events/$CID" -H 'Content-Type: application/json' \
  -d "{\"expected_version\":$VER,\"summary\":\"用旧版本改应该冲突\"}")
[ "$C" = 409 ] && ok "旧 expected_version → 409 冲突" || bad "stale version" "$C" "$(body)"
C=$(code -X DELETE "$BASE/api/manual-events/$CID" -H 'Content-Type: application/json' \
  -d "{\"expected_version\":$((VER+1))}")
[ "$C" = 200 ] && ok "DELETE 事件 200" || bad "delete event" "$C" "$(body)"

say "== 3. 边 增/查/删 =="
C=$(code -X POST "$BASE/api/edges" -H 'Content-Type: application/json' \
  -d "{\"from_type\":\"event\",\"from_id\":\"$CID\",\"to_type\":\"event\",\"to_id\":\"$CID\",\"relation\":\"related_to\"}")
[ "$C" = 400 ] && ok "自环边被拒绝 400" || bad "self loop" "$C" "$(body)"
# 取两个真实节点建边
IDS=$(${PY:-python3} - <<PYEOF
import json, urllib.request
def get(u):
    return json.load(urllib.request.urlopen("$BASE" + u))
ents = get("/api/nodes/entity?limit=5")["nodes"]
evs = get("/api/nodes/event?limit=5")["nodes"]
if ents and evs:
    print(ents[0]["id"], evs[0]["id"])
else:
    print("", "")
PYEOF
)
EID2=$(echo "$IDS" | awk '{print $1}'); VID2=$(echo "$IDS" | awk '{print $2}')
if [ -n "$EID2" ] && [ -n "$VID2" ] && [ "$EID2" != "" ]; then
  C=$(code -X POST "$BASE/api/edges" -H 'Content-Type: application/json' \
    -d "{\"from_type\":\"entity\",\"from_id\":\"$EID2\",\"to_type\":\"event\",\"to_id\":\"$VID2\",\"relation\":\"related_to\",\"note\":\"冒烟测试边\"}")
  RID=$(body | ${PY:-python3} -c 'import json,sys; print(json.load(sys.stdin).get("relation_id",""))' || echo "")
  if [ "$C" = 201 ] && [ -n "$RID" ]; then
    ok "POST 边 201"
    C=$(code "$BASE/api/edges?from=$EID2"); [ "$C" = 200 ] && ok "GET 边列表 200" || bad "list edges" "$C" "$(body)"
    C=$(code -X DELETE "$BASE/api/edges/$RID"); [ "$C" = 200 ] && ok "DELETE 边 200" || bad "delete edge" "$C" "$(body)"
  else
    bad "create edge" "$C" "$(body)"
  fi
else
  say "  - 跳过边测试（库为空，先运行流水线）"
fi

say "== 4. 图谱/事实/时间线/变化/检索 =="
for EP in "/api/graph" "/api/facts" "/api/timeline" "/api/changes" "/api/stats" "/api/decisions" "/api/events?limit=5"; do
  C=$(code "$BASE$EP"); [ "$C" = 200 ] && ok "GET $EP 200" || bad "$EP" "$C" "$(body)"
done
C=$(code -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":"调价"}')
[ "$C" = 200 ] && ok "POST /api/search 200" || bad "search" "$C" "$(body)"

say "== 5. 问答（SSE 流） =="
SSE=$(curl -s -N --max-time 150 -X POST "$BASE/api/answer" -H 'Content-Type: application/json' \
  -d '{"query":"有哪些公司发生了价格调整？"}' 2>/dev/null | head -c 20000 || true)
echo "$SSE" | grep -q '"type": "evidence"' && ok "answer SSE 先吐证据包" || bad "answer evidence" "-" "未见 evidence 帧"
echo "$SSE" | grep -q '"type": "done"' && ok "answer SSE 正常收尾" || bad "answer done" "-" "未见 done 帧（可能超时，单独复验）"

say ""
say "===== 冒烟结果: 通过 $PASS / 失败 $FAIL ====="
[ "$FAIL" = 0 ]
