# -*- coding: utf-8 -*-
"""URL 规范化唯一实现（ingest 与语料整理器共用，DATA_DESIGN clean-v1 规则）。

统一规则保证：同一 URL 无论从 v2 原始批还是 clean 清洗层进入，
document_id (由 source_id + norm_url 派生) 完全一致 —— 双轨零重复入库。
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# 广告/分享追踪参数（去 query 时剥离）
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "spm", "from", "share_token", "chksm", "scene", "pos", "shareid",
    "src", "sid", "tt_from", "group_id",
}


def norm_url(url: str) -> str:
    """规范化 URL：小写 scheme/host、去 tracking 参数、去 fragment、去尾斜杠。"""
    try:
        sp = urlsplit((url or "").strip())
        q = [(k, v) for k, v in parse_qsl(sp.query, keep_blank_values=False)
             if k.lower() not in TRACKING_PARAMS]
        path = sp.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return urlunsplit((sp.scheme.lower() or "https", sp.netloc.lower(), path,
                           urlencode(q), ""))
    except ValueError:
        return (url or "").strip()
