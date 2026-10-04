# -*- coding: utf-8 -*-
"""HTTP API 服务：REST + SSE 实时观测 + 静态前端（纯标准库，无框架依赖）。

API 面（对齐 04 §16.1 的单机子集）：
  GET  /api/health | /api/stats
  GET  /api/events?after_id=&stage=&level=&kind=&limit=    可观测事件（增量）
  GET  /api/stream                                          SSE 实时事件流
  GET  /api/decisions                                       归并决策列表
  GET  /api/graph?focus=&hops=                              图谱视图
  GET  /api/nodes/{type} ?q=&limit=                          节点列表
  GET  /api/entities/{id} | /api/events/cluster/{id}         节点详情
  POST /api/entities           PATCH/DELETE /api/entities/{id}
  POST /api/edges              PATCH/DELETE /api/edges/{id}
  POST /api/manual-events      PATCH/DELETE /api/manual-events/{id}   人工事件 CRUD
  POST /api/cluster-ops        move_mention | split_cluster | merge_clusters | retract_assertion
  GET  /api/facts?slot=&entity=&valid_as_of=&known_as_of=   双时间事实
  GET  /api/timeline?entity=&process=
  GET  /api/changes            变化卡
  POST /api/search {query}
  POST /api/answer {query}     → SSE 流（evidence → tokens → done）
  POST /api/ingest {items}     外部提交资料
  POST /api/replay             重跑流水线（缓存复用，可复现）
  POST /api/pipeline/run       在线触发流水线（异步线程）
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, config as cfg, graph_service as G, llm, query_service as Q
from . import observe
from .pipeline import runner as R
from .store import db, queue
from .util import now_iso

DB_LOCK = threading.Lock()  # 串行化写事务，与 SQLite WAL 配合


def _db():
    conn = db.connect(cfg.DB_PATH)
    db.init_db(conn)
    return conn


class Handler(BaseHTTPRequestHandler):
    server_version = f"IntelIO/{__version__}"

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):  # 静默访问日志（观测走 pipeline_event）
        pass

    def _send(self, code: int, obj, headers: dict | None = None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, e: Exception):
        if isinstance(e, G.Conflict):
            self._send(409, {"error": str(e), "code": "version_conflict"})
        elif isinstance(e, G.NotFound):
            self._send(404, {"error": str(e), "code": "not_found"})
        elif isinstance(e, G.BadRequest):
            self._send(400, {"error": str(e), "code": "bad_request"})
        else:
            self._send(500, {"error": str(e), "code": "internal", "at": now_iso()})

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PATCH,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            self._route_get()
        except Exception as e:  # noqa: BLE001
            self._err(e)

    def do_POST(self):
        try:
            self._route_post()
        except Exception as e:  # noqa: BLE001
            self._err(e)

    def do_PATCH(self):
        try:
            self._route_patch()
        except Exception as e:  # noqa: BLE001
            self._err(e)

    def do_DELETE(self):
        try:
            self._route_delete()
        except Exception as e:  # noqa: BLE001
            self._err(e)

    # ------------------------------------------------------------------
    def _route_get(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = u.path
        conn = _db()
        try:
            if path in ("/", "/index.html"):
                self._static("index.html", "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._static_file(path[len("/static/"):])
            elif path == "/api/health":
                self._send(200, {"ok": True, "version": __version__,
                                 "models": llm.health(), "time": now_iso()})
            elif path == "/api/stats":
                with DB_LOCK:
                    self._send(200, {"stats": R.stats(conn), "queues": queue.depths(conn),
                                     "time": now_iso()})
            elif path == "/api/events":
                rows = observe.recent(
                    conn, after_id=int(qs.get("after_id", 0)),
                    limit=min(int(qs.get("limit", 200)), 1000),
                    stage=qs.get("stage"), level=qs.get("level"), kind=qs.get("kind"))
                self._send(200, {"events": rows})
            elif path == "/api/decisions":
                rows = conn.execute(
                    "SELECT d.*, m.frame_text, c.summary AS cluster_summary, c.state AS cluster_state "
                    "FROM resolution_decision d "
                    "LEFT JOIN event_mention m ON m.mention_id=d.mention_id "
                    "LEFT JOIN event_cluster c ON c.cluster_id=d.target_cluster_id "
                    "ORDER BY d.created_at DESC LIMIT ?",
                    (min(int(qs.get("limit", 100)), 500),)).fetchall()
                self._send(200, {"decisions": [dict(r) for r in rows]})
            elif path == "/api/graph":
                self._send(200, G.graph_view(
                    conn, focus=qs.get("focus"), hops=int(qs.get("hops", 1)),
                    limit=int(qs.get("limit", 200)), mode=qs.get("mode")))
            elif m := re.match(r"^/api/nodes/(entity|event|process|assertion)$", path):
                self._send(200, {"nodes": G.list_nodes(
                    conn, m.group(1), q=qs.get("q"), limit=int(qs.get("limit", 100)),
                    offset=int(qs.get("offset", 0)))})
            elif m := re.match(r"^/api/entities/([^/]+)$", path):
                self._send(200, G.get_entity(conn, m.group(1)))
            elif m := re.match(r"^/api/events/cluster/([^/]+)$", path):
                self._send(200, G.get_event(conn, m.group(1),
                                            version=int(qs["v"]) if qs.get("v") else None))
            elif path == "/api/edges":
                self._send(200, {"edges": G.list_edges(
                    conn, from_id=qs.get("from"), to_id=qs.get("to"),
                    relation=qs.get("relation"), limit=int(qs.get("limit", 200)))})
            elif path == "/api/facts":
                self._facts(conn, qs)
            elif path == "/api/timeline":
                self._send(200, {"timeline": Q.timeline(
                    conn, entity_id=qs.get("entity"), process_id=qs.get("process"),
                    event_type=qs.get("type"), limit=int(qs.get("limit", 100)))})
            elif path == "/api/changes":
                rows = conn.execute(
                    "SELECT ch.*, e.canonical_name AS subject FROM change_record ch "
                    "LEFT JOIN entity e ON e.entity_id=ch.subject_entity_id "
                    "ORDER BY ch.created_at DESC LIMIT ?",
                    (min(int(qs.get("limit", 100)), 500),)).fetchall()
                self._send(200, {"changes": [dict(r) for r in rows]})
            elif path == "/api/pipeline/status":
                rows = conn.execute(
                    "SELECT stage, status, MAX(started_at) AS started_at, "
                    "MAX(finished_at) AS finished_at, detail FROM pipeline_run "
                    "GROUP BY stage, status ORDER BY started_at").fetchall()
                stages: dict = {}
                for r in rows:  # 同一阶段取最近一次运行
                    stages[r["stage"]] = dict(r)
                self._send(200, {"stages": stages})
            elif path == "/api/search-events":
                q = qs.get("q", "").strip()
                if not q:
                    raise G.BadRequest("q 不能为空")
                analysis = Q.analyze_query(conn, q)
                ret = Q.retrieve_events(conn, q, analysis)
                self._send(200, {"analysis": {
                    "entities": [e["canonical_name"] for e in analysis["entities"]],
                    "family_size": len(analysis["family_ids"]),
                    "win_from": analysis["win_from"], "win_to": analysis["win_to"],
                    "intent": analysis["intent"]}, "retrieval": ret})
            elif m := re.match(r"^/api/entity-history/([^/]+)$", path):
                eid = m.group(1)
                ent = conn.execute("SELECT canonical_name FROM entity WHERE entity_id=? "
                                   "AND deleted_at IS NULL", (eid,)).fetchone()
                if ent is None:
                    raise G.NotFound("实体不存在")
                from .query_service import _slot_history
                facts = _slot_history(conn, [eid])
                evs = Q.timeline(conn, entity_id=eid, limit=200)
                self._send(200, {"entity": dict(ent), "facts": facts, "events": evs})
            elif path == "/api/manifest":
                row = conn.execute("SELECT value FROM meta WHERE key='last_run_manifest'"
                                   ).fetchone()
                self._send(200, {"manifest": json.loads(row["value"]) if row else None})
            else:
                self._send(404, {"error": f"unknown path {path}"})
        finally:
            conn.close()

    def _facts(self, conn, qs):
        slots = []
        if qs.get("slot"):
            slots = [qs["slot"]]
        elif qs.get("entity"):
            for r in conn.execute("SELECT DISTINCT slot_key FROM assertion WHERE "
                                  "subject_entity_id=?", (qs["entity"],)).fetchall():
                slots.append(r["slot_key"])
        else:
            for r in conn.execute("SELECT DISTINCT slot_key FROM slot_selection_history "
                                  "ORDER BY slot_key LIMIT 50").fetchall():
                slots.append(r["slot_key"])
        facts = [Q.current_fact(conn, s, valid_as_of=qs.get("valid_as_of"),
                                known_as_of=qs.get("known_as_of")) for s in slots]
        self._send(200, {"facts": facts})

    # ------------------------------------------------------------------
    def _route_post(self):
        u = urlparse(self.path)
        path = u.path
        body = self._body()
        conn = _db()
        try:
            if path == "/api/answer":
                self._answer(conn, body)
                return
            if path == "/api/search":
                query = (body.get("query") or "").strip()
                if not query:
                    raise G.BadRequest("query 不能为空")
                self._send(200, Q.search(conn, query))
                return
            if path == "/api/ingest":
                from .pipeline import ingest
                ids = []
                with DB_LOCK:
                    for item in body.get("items", []):
                        with conn:
                            ids.append(ingest.ingest_item(conn, item))
                self._send(200, {"ingested": [i for i in ids if i]})
                return
            if path == "/api/pipeline/run" or path == "/api/replay":
                stages = body.get("stages")
                use_cache = bool(body.get("use_cache", True))
                th = threading.Thread(target=self._run_pipeline_bg,
                                      args=(stages, use_cache), daemon=True)
                th.start()
                self._send(202, {"started": True, "stages": stages or R.STAGES})
                return
            if path == "/api/entities":
                with DB_LOCK, conn:
                    out = G.create_entity(conn, body.get("name", ""),
                                          subtype=body.get("type", "other"),
                                          aliases=body.get("aliases"),
                                          note=body.get("note"))
                self._send(201, out)
                return
            if path == "/api/edges":
                with DB_LOCK, conn:
                    out = G.create_edge(conn, body.get("from_type", ""),
                                        body.get("from_id", ""), body.get("to_type", ""),
                                        body.get("to_id", ""), body.get("relation", ""),
                                        note=body.get("note"),
                                        evidence=body.get("evidence"))
                self._send(201, out)
                return
            if path == "/api/manual-events":
                with DB_LOCK, conn:
                    out = G.create_event(
                        conn, event_type=body.get("event_type", "other"),
                        title=body.get("title", ""),
                        event_time_lower=body.get("event_time_lower"),
                        event_time_upper=body.get("event_time_upper"),
                        entity_ids=body.get("entity_ids"), process_id=body.get("process_id"),
                        note=body.get("note"))
                self._send(201, out)
                return
            if path == "/api/cluster-ops":
                self._cluster_ops(conn, body)
                return
            self._send(404, {"error": f"unknown path {path}"})
        finally:
            conn.close()

    def _cluster_ops(self, conn, body):
        op = body.get("op")
        with DB_LOCK, conn:
            if op == "move_mention":
                out = G.move_mention(conn, body.get("mention_id", ""),
                                     body.get("target_cluster_id", ""), reason=body.get("reason", ""),
                                     expected_source_version=body.get("expected_source_version"),
                                     expected_target_version=body.get("expected_target_version"))
            elif op == "split_cluster":
                out = G.split_cluster(conn, body.get("cluster_id", ""),
                                      body.get("mention_ids") or [], reason=body.get("reason", ""),
                                      expected_version=int(body.get("expected_version", 0)))
            elif op == "merge_clusters":
                out = G.merge_clusters(conn, body.get("source_ids") or [],
                                       body.get("target_cluster_id", ""), reason=body.get("reason", ""),
                                       expected_version=int(body.get("expected_version", 0)))
            elif op == "retract_assertion":
                out = G.retract_assertion(conn, body.get("assertion_id", ""),
                                          basis=body.get("basis", ""))
            else:
                raise G.BadRequest(f"未知操作: {op}")
        self._send(200, out)

    def _run_pipeline_bg(self, stages, use_cache):
        conn = _db()
        try:
            with DB_LOCK:
                R.run_pipeline(conn, stages=stages, use_cache=use_cache)
        except Exception as e:  # noqa: BLE001
            try:
                conn2 = _db()
                observe.emit(conn2, "pipeline", f"流水线失败: {e}", level="error",
                             kind="pipeline.error")
                conn2.close()
            except Exception:  # noqa: BLE001
                pass
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _route_patch(self):
        u = urlparse(self.path)
        path = u.path
        body = self._body()
        conn = _db()
        try:
            if m := re.match(r"^/api/entities/([^/]+)$", path):
                with DB_LOCK, conn:
                    out = G.update_entity(conn, m.group(1), name=body.get("name"),
                                          add_aliases=body.get("add_aliases"),
                                          remove_alias_ids=body.get("remove_alias_ids"),
                                          note=body.get("note"),
                                          expected_updated=body.get("expected_updated"))
                self._send(200, out)
                return
            if m := re.match(r"^/api/edges/([^/]+)$", path):
                with DB_LOCK, conn:
                    out = G.update_edge(conn, m.group(1), note=body.get("note"),
                                        valid_from=body.get("valid_from"),
                                        valid_to=body.get("valid_to"))
                self._send(200, out)
                return
            if m := re.match(r"^/api/manual-events/([^/]+)$", path):
                if "expected_version" not in body:
                    raise G.BadRequest("缺少 expected_version（乐观锁）")
                with DB_LOCK, conn:
                    out = G.update_event(conn, m.group(1), **body)
                self._send(200, out)
                return
            self._send(404, {"error": f"unknown path {path}"})
        finally:
            conn.close()

    def _route_delete(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = u.path
        body = self._body()
        conn = _db()
        try:
            if m := re.match(r"^/api/entities/([^/]+)$", path):
                with DB_LOCK, conn:
                    out = G.delete_entity(conn, m.group(1),
                                          expected_updated=body.get("expected_updated"))
                self._send(200, out)
                return
            if m := re.match(r"^/api/edges/([^/]+)$", path):
                with DB_LOCK, conn:
                    out = G.delete_edge(conn, m.group(1))
                self._send(200, out)
                return
            if m := re.match(r"^/api/manual-events/([^/]+)$", path):
                if "expected_version" not in body:
                    raise G.BadRequest("缺少 expected_version（乐观锁）")
                with DB_LOCK, conn:
                    out = G.delete_event(conn, m.group(1),
                                         expected_version=int(body["expected_version"]),
                                         force=bool(body.get("force")))
                self._send(200, out)
                return
            self._send(404, {"error": f"unknown path {path}"})
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _answer(self, conn, query_body):
        """SSE：evidence → token... → done。"""
        query = (query_body.get("query") or "").strip()
        if not query:
            self._send(400, {"error": "query 不能为空"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            for ev in Q.answer_stream(conn, query):
                data = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, name: str, ctype: str):
        p = cfg.WEB_DIR / name
        if not p.exists():
            self._send(404, {"error": f"missing {name}"})
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _static_file(self, name: str):
        ct = {"js": "application/javascript; charset=utf-8",
              "css": "text/css; charset=utf-8",
              "html": "text/html; charset=utf-8",
              "svg": "image/svg+xml", "png": "image/png",
              "ico": "image/x-icon"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
        self._static(name, ct)


class SSEHandler(Handler):
    """/api/stream：轮询 pipeline_event 增量推送。"""

    def _route_get(self):
        if urlparse(self.path).path != "/api/stream":
            return super()._route_get()
        qs = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        cursor = int(qs.get("after_id", 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        conn = _db()
        try:
            idle = 0
            while True:
                rows = observe.recent(conn, after_id=cursor, limit=100)
                if rows:
                    rows.reverse()
                    for r in rows:
                        cursor = r["event_id"]
                        data = json.dumps(r, ensure_ascii=False)
                        self.wfile.write(f"id: {cursor}\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    idle = 0
                else:
                    idle += 1
                    if idle % 15 == 0:  # 心跳
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    time.sleep(0.7)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()


def serve(host: str = "127.0.0.1", port: int = 8600) -> None:
    cfg.ensure_dirs()
    conn = _db()
    conn.close()
    httpd = ThreadingHTTPServer((host, port), SSEHandler)
    print(f"[intel-server] http://{host}:{port}  (version {__version__})")
    observe_conn = _db()
    observe.emit(observe_conn, "server", f"观测服务启动: http://{host}:{port}",
                 kind="server.start")
    observe_conn.close()
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()
    serve(args.host, args.port)
