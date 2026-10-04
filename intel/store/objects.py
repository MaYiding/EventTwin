# -*- coding: utf-8 -*-
"""内容寻址原文对象库（对象存储的单机映射）：按 sha256 存取不可变原文。"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..util import sha256_bytes


def put(content: bytes) -> tuple[str, Path]:
    """保存原文，返回 (content_hash, 路径)。同哈希幂等。"""
    config.ensure_dirs()
    h = sha256_bytes(content)
    rel = Path(h[:2]) / Path(h[2:4]) / h
    path = config.OBJECTS_DIR / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)  # 原子落盘，避免半写文件
    return h, path


def get(content_hash: str) -> bytes | None:
    path = config.OBJECTS_DIR / content_hash[:2] / content_hash[2:4] / content_hash
    if not path.exists():
        return None
    return path.read_bytes()
