# -*- coding: utf-8 -*-
"""模型客户端（v3.1 三件套架构）：Chat(Qwen3.8-27B) + Embedding(Qwen3-Embedding-8B)
+ Jev(TypeSafe System One 判定模型)。服务：10.10.21.216（902 容器，vLLM）。

分工依据 2026-09-24 五模型评测（eval-classification-20260923，v3 精确标注 262 条）：
- qwen3.8-27b：生成类任务（提及抽取 / 问答）；判定类任务上生成模型普遍
  "宽容判官"（8B 细粒度伪造召回 0），只做生成不做判别；
- Jev：全部判定类任务（同事件判别 / 候选同事件分布 / 检索相关性重排）——
  事件核对 1.0、实体核对 0.966、层级分类 0.931，五个模型中唯一可担判定职责；
- qwen3-embedding-8b：向量（4096 维，L2 归一化），只用于召回与 S_semantic 特征
  ——不参与判定（评测：embedding 判定面"全拒型"）；reranker / Laya 不再使用。

要点：
- 新 vLLM 服务不识别 /no_think 软开关：思考关闭改走 chat_template_kwargs.
  enable_thinking=false（Qwen3+vLLM 标准口径）；另对泄漏的裸 </think> 做兜底截断；
- 所有模型调用（chat / embed / decisions）走同一 LLM 缓存（sha256(模型+请求体)
  命中即复用），保证【过程可复现】：重放同一语料时结果逐字一致，且不重复消耗推理；
  embedding 另有独立向量缓存（按 文本哈希，避免重复向量请求）；
- 网络错误/5xx 超时重试（Jev 另对 429/529 退避）；JSON 输出解析失败自动追加重试一次。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time

import requests

from . import config as cfg
from .util import canonical_json, extract_json, sha256_text

_cfg = None
_lock = threading.Lock()


def _conf() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = cfg.load_config()["models"]
    return _cfg


# ---------------------------------------------------------------------------
# .env 加载（OR_KEY 等凭据：环境变量优先，其次 Code 根目录 .env，不入库不入 git）
# ---------------------------------------------------------------------------

_env_loaded = False
_env_lock = threading.Lock()


def _load_env() -> None:
    global _env_loaded
    if _env_loaded:
        return
    with _env_lock:                      # 并发首载竞态：标志先行会让后到线程读到空环境
        if _env_loaded:
            return
        env_path = cfg.ROOT / ".env"
        if not env_path.exists():
            _env_loaded = True
            return
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        except OSError:
            pass
        _env_loaded = True


# ---------------------------------------------------------------------------
# LLM 调用缓存（跨进程共享，data/state/llm_cache.db）
# ---------------------------------------------------------------------------

def _cache_conn() -> sqlite3.Connection:
    cfg.ensure_dirs()
    conn = sqlite3.connect(str(cfg.LLM_CACHE_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache ("
        "cache_key TEXT PRIMARY KEY, model_id TEXT, stage TEXT, "
        "response_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def cache_get(key: str) -> dict | None:
    try:
        conn = _cache_conn()
        row = conn.execute("SELECT response_json FROM llm_cache WHERE cache_key=?", (key,)).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None
    except sqlite3.Error:
        return None


def cache_put(key: str, model_id: str, stage: str, response: dict) -> None:
    from .util import now_iso
    try:
        conn = _cache_conn()
        conn.execute(
            "INSERT OR IGNORE INTO llm_cache(cache_key, model_id, stage, response_json, created_at) "
            "VALUES (?,?,?,?,?)",
            (key, model_id, stage, canonical_json(response), now_iso()),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def cache_enabled() -> bool:
    return bool(cfg.load_config()["pipeline"]["llm_cache"])


# ---------------------------------------------------------------------------
# Chat（qwen3-8b，生成类任务：抽取 / 问答）
# ---------------------------------------------------------------------------

class LLMError(Exception):
    pass


def chat(messages: list[dict], *, stage: str = "chat", temperature: float = 0.7,
         max_tokens: int = 2048, no_think: bool = True, json_mode: bool = False,
         use_cache: bool = True) -> str:
    """调用 qwen3.8-27b，返回正文文本。json_mode=True 时同时返回解析后的对象。"""
    c = _conf()
    msgs = [dict(m) for m in messages]
    req = {
        "model": c["chat_model"],
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if no_think:
        # vLLM 部署的 Qwen3 不识别 /no_think 软开关，标准口径是模板参数关思考
        # （实测 2026-09-29：enable_thinking=false 输出干净正文，/no_think 会被复述）
        req["chat_template_kwargs"] = {"enable_thinking": False}
    key = sha256_text(canonical_json({
        "model": c["chat_model"], "messages": msgs, "temperature": temperature,
        "max_tokens": max_tokens,
    }))
    if use_cache and cache_enabled():
        hit = cache_get(key)
        if hit is not None:
            return hit["content"]
    body = _request_with_retry(
        c["chat_base_url"] + "/chat/completions", req, timeout=c["chat_timeout_seconds"],
        retries=c["max_retries"], stage=stage,
    )
    content = _extract_content(body)
    reasoning = _extract_reasoning(body)
    if use_cache:
        cache_put(key, c["chat_model"], stage, {"content": content, "reasoning": reasoning})
    return content


def chat_json(messages: list[dict], *, stage: str = "chat", temperature: float = 0.0,
              max_tokens: int = 3000, use_cache: bool = True):
    """要求 JSON 输出并稳健解析。

    失败处理：疑似截断（输出没有正常收尾）→ 加倍 max_tokens 重试一次；
    否则追加"只输出 JSON"纠错提示重试一次；仍失败抛 LLMError。
    """
    content = chat(messages, stage=stage, temperature=temperature, max_tokens=max_tokens,
                   no_think=True, use_cache=use_cache)
    obj = extract_json(content)
    if obj is not None:
        return obj
    stripped = content.strip().rstrip("`")
    truncated = not stripped.endswith("}") and not stripped.endswith("]")
    if truncated and max_tokens < 8192:
        content2 = chat(messages, stage=stage + ".wider", temperature=temperature,
                        max_tokens=min(max_tokens * 2, 8192), no_think=True, use_cache=use_cache)
        obj = extract_json(content2)
        if obj is not None:
            return obj
        content = content2
    retry_messages = messages + [
        {"role": "user", "content": "你上一条回复无法解析为 JSON。请只输出一个合法的 JSON，"
                                    "不要任何解释、代码围栏或其他文字。输出务必完整收尾。"}
    ]
    content2 = chat(retry_messages, stage=stage + ".retry", temperature=0.0,
                    max_tokens=max(2048, max_tokens), no_think=True, use_cache=use_cache)
    obj = extract_json(content2)
    if obj is None:
        raise LLMError(f"[{stage}] 模型未返回可解析 JSON: {content[:300]}")
    return obj


def _extract_content(body: dict) -> str:
    try:
        msg = body["choices"][0]["message"]
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            # 剥离泄漏出的思考：成对 <think>...</think> 整段删除；
            # 只剩裸 </think>（开头无 <think>）时取其后正文（vLLM 未配 reasoning parser 的兜底）
            c = re.sub(r"<think>.*?</think>", "", c, flags=re.S)
            if "</think>" in c:
                c = c.split("</think>")[-1]
            return c.strip()
        # 兼容 content 为空、正文放在 reasoning 之后段落的情形
        rc = msg.get("reasoning_content") or ""
        if rc and not c:
            raise LLMError("模型只输出了思考内容没有正式回答（max_tokens 可能不足）")
        return c or ""
    except (KeyError, IndexError) as e:
        raise LLMError(f"chat 响应结构异常: {e}") from e


def _extract_reasoning(body: dict) -> str:
    try:
        return body["choices"][0]["message"].get("reasoning_content") or ""
    except (KeyError, IndexError):
        return ""


# ---------------------------------------------------------------------------
# Embedding（qwen3-embedding-8b，只用于召回与 S_semantic 特征，不参与判定）
# ---------------------------------------------------------------------------

def embed(texts: list[str], *, stage: str = "embed", use_cache: bool = True):
    """批量向量化（qwen3-embedding-8b），返回 float32 矩阵（已 L2 归一化）。

    向量按 (模型, 文本哈希) 独立缓存 —— 重放同一语料零推理且逐字一致；
    批间并行（结果按 index 归位，向量只取决于文本，确定性不变）。
    """
    import numpy as np
    from .store import embedding_cache as EC
    c = _conf()
    clean = [t if t.strip() else " " for t in texts]
    cached: dict[int, "np.ndarray"] = {}
    if use_cache:
        for i, t in enumerate(clean):
            hit = EC.get(c["embed_model"], t)
            if hit is not None:
                cached[i] = hit
    out: list = [None] * len(clean)
    todo_idx = [i for i in range(len(clean)) if i not in cached]
    B = 16

    def _embed_batch(chunk: list[int]) -> None:
        req = {"model": c["embed_model"], "input": [clean[i] for i in chunk]}
        body = _request_with_retry(
            c["embed_base_url"] + "/embeddings", req, timeout=c.get("embed_timeout_seconds", 120),
            retries=c["max_retries"], stage=stage,
        )
        try:
            data = sorted(body["data"], key=lambda d: d.get("index", 0))
            for d, i in zip(data, chunk):
                vec = np.asarray(d["embedding"], dtype=np.float32)
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
                out[i] = vec
                if use_cache:
                    EC.put(c["embed_model"], clean[i], vec)
        except (KeyError, TypeError) as e:
            raise LLMError(f"embedding 响应结构异常: {e}") from e

    from concurrent.futures import ThreadPoolExecutor
    batches = [todo_idx[j:j + B] for j in range(0, len(todo_idx), B)]
    if batches:
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as ex:
            futs = [ex.submit(_embed_batch, b) for b in batches]
            for f in futs:
                f.result()  # 首个异常向上抛
    for i, v in cached.items():
        out[i] = v
    mat = np.stack(out)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Jev（TypeSafe System One，判定类任务：同事件判别 / 相关性分布）
# ---------------------------------------------------------------------------

def decide(state: dict, questions: dict, *, stage: str = "jev",
           use_cache: bool = True) -> dict:
    """Jev 决策调用（systemone 协议：state + questions → answers）。

    questions 支持 noul（是否判断）/ choice（单选 + 全分布 probabilities）/
    score（等级评分），一次请求可带多个问题（评测实测 200 问/请求无压力）。
    结果按 sha256(模型+请求体) 落 LLM 缓存 —— 与 chat 同一套可复现机制。
    """
    _load_env()
    c = _conf()
    payload = {"model": c["jev_model"], "state": state, "questions": questions}
    key = sha256_text(canonical_json({"kind": "decisions", **payload}))
    if use_cache and cache_enabled():
        hit = cache_get(key)
        if hit is not None and hit.get("answers") is not None:
            return hit["answers"]
    url = c.get("jev_base_url", "https://openrouter.ai/api/alpha/decisions")
    token = os.environ.get("OR_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not token:
        raise LLMError(f"[{stage}] 缺少 OR_KEY（环境变量或 {cfg.ROOT}/.env）")
    body = _request_with_retry(
        url, payload, timeout=c.get("jev_timeout_seconds", 240),
        retries=c["max_retries"], stage=stage,
        headers={"Authorization": f"Bearer {token}"}, retry_on_429=True)
    answers = body.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise LLMError(f"[{stage}] Jev 响应缺少 answers: {str(body)[:200]}")
    if use_cache:
        cache_put(key, c["jev_model"], stage, {"answers": answers})
    return answers


def jev_choice_rank(query: str, documents: list[str], *, instructions: str | None = None,
                    none_desc: str = "以上候选均与查询不相关",
                    stage: str = "jev_rank", use_cache: bool = True) -> list[float]:
    """Jev 相关性/同事件分布（替代 qwen3-reranker 的排序角色）。

    单次 choice：每个文档一个选项 + none 兜底项，返回各文档概率（与入参顺序对齐）。
    概率不重归一 —— none 的概率质量代表"都不匹配"，此时各候选自然全低分，
    调用方（低分新建 / 低相关过滤）依赖的正是这一语义。
    choice 选项上限 255，超出须由调用方先确定性预截断。
    """
    c = _conf()
    cap = int(c.get("jev_doc_chars", 300))
    docs = [str(d)[:cap] for d in documents]
    criteria = {f"d{i}": d for i, d in enumerate(docs)}
    criteria["none_of_them"] = none_desc
    state = {"查询": query[:2000], "候选": {k: v for k, v in criteria.items()
                                           if k != "none_of_them"}}
    questions = {"rank": {
        "type": "choice",
        "instructions": instructions or (
            "判断 `查询` 与 `候选` 中哪一项最相关：选出能直接回答或对应 `查询` "
            "所描述内容的一项；仅主题相近但对象、时间或动作不同的不算相关。"),
        "criteria": criteria}}
    answers = decide(state, questions, stage=stage, use_cache=use_cache)
    a = answers.get("rank") or {}
    probs = a.get("probabilities") or {}
    return [float(probs.get(f"d{i}", 0.0)) for i in range(len(docs))]


# ---------------------------------------------------------------------------
# 公共 HTTP 重试
# ---------------------------------------------------------------------------

def _request_with_retry(url: str, payload: dict, *, timeout: float, retries: int,
                        stage: str, headers: dict | None = None,
                        retry_on_429: bool = False) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload,
                                 timeout=(10, timeout),  # 连接 10s 快速失败，读取按配置
                                 headers={"Content-Type": "application/json", **(headers or {})})
            if resp.status_code >= 500 or (retry_on_429 and resp.status_code == 429):
                raise requests.HTTPError(f"{resp.status_code} {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 统一重试
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (2 ** attempt))  # 429/529 时退避加剧
    raise LLMError(f"[{stage}] 模型服务调用失败（{url}）: {last_err}") from last_err


def health() -> dict:
    """三个模型端点健康检查：chat / embed（vLLM /models）+ jev（OpenRouter key 校验）。"""
    c = _conf()
    out = {}
    for name, base in (("chat", c["chat_base_url"]), ("embed", c["embed_base_url"])):
        try:
            r = requests.get(base + "/models", timeout=5)
            out[name] = {"ok": r.status_code == 200}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": str(e)[:200]}
    _load_env()
    token = os.environ.get("OR_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not token:
        out["jev"] = {"ok": False, "error": "缺少 OR_KEY"}
        return out
    try:
        r = requests.get("https://openrouter.ai/api/v1/key", timeout=8,
                         headers={"Authorization": f"Bearer {token}"})
        out["jev"] = {"ok": r.status_code == 200}
    except Exception as e:  # noqa: BLE001
        out["jev"] = {"ok": False, "error": str(e)[:200]}
    return out
