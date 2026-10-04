# -*- coding: utf-8 -*-
"""配置加载与路径管理。配置文件 config.json 与代码同目录，运行时状态在 data/state。"""
from __future__ import annotations

import json
from pathlib import Path

# 代码根目录：intel 包的上一级（即 Code/）
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
STATE_DIR = ROOT / "data" / "state"
OBJECTS_DIR = STATE_DIR / "objects"
CORPUS_DIR = ROOT / "data" / "corpus" / "incoming"
WEB_DIR = Path(__file__).resolve().parent / "web"

DB_PATH = STATE_DIR / "intel.db"
LLM_CACHE_PATH = STATE_DIR / "llm_cache.db"
EMBED_CACHE_PATH = STATE_DIR / "embedding_cache.db"


def ensure_dirs() -> None:
    for p in (STATE_DIR, OBJECTS_DIR, CORPUS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
