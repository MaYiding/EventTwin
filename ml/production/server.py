#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同事件判别器推理服务（三模型集成为一个 API）。

用法：
  python3 -m ml.production.server --port 8601
  curl -X POST localhost:8601/judge \\
    -H 'Content-Type: application/json' \\
    -d '{"events_a": [{"frame": "...", "type": "price_change"}],
         "events_b": [{"frame": "...", "type": "price_change"}]}'
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROD_DIR = Path(__file__).parent


class JudgeServer(BaseHTTPRequestHandler):
    judge = None

    def do_POST(self):
        if self.path == "/judge":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            pairs = list(zip(body["events_a"], body["events_b"]))
            t0 = time.time()
            scores = self.judge.judge_batch(pairs)
            dt = time.time() - t0
            resp = json.dumps({
                "scores": [round(s, 4) for s in scores],
                "decisions": ["same" if s >= 0.5 else "different" for s in scores],
                "latency_ms": round(dt * 1000, 1),
                "model": "ensemble(v15+v8+v10a+TTA)",
            }, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--models-dir", default=str(PROD_DIR / "models"))
    args = ap.parse_args()

    from ml.judges import EnsembleJudge
    JudgeServer.judge = EnsembleJudge(
        models={
            "v15": str(Path(args.models_dir) / "v15_4b_merged"),
            "v8": str(Path(args.models_dir) / "v8"),
            "v10a": str(Path(args.models_dir) / "v10a"),
        },
        weights={"v15": 0.8, "v8": 0.1, "v10a": 0.1},
        temperature=0.5,
        tta=True,
    )
    server = HTTPServer(("0.0.0.0", args.port), JudgeServer)
    print(f"判别器服务启动: http://0.0.0.0:{args.port}")
    print(f"  POST /judge   → 集成判分（15ms/对 GPU）")
    print(f"  GET  /health  → 健康检查")
    server.serve_forever()


if __name__ == "__main__":
    main()
