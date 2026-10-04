# -*- coding: utf-8 -*-
"""通用工具：确定性 ID、内容哈希、时间解析、引文定位、JSON 清洗。

设计原则（对齐架构文档 04 §3.3 / §5.2 / §11.3）：
- ID 稳定：document/mention/cluster 等 ID 由 uuid5 派生，同输入重放得到同 ID；
- 引文可信：模型给出的 quote 必须能在规范正文中定位到字符区间，定位不到即丢弃；
- 五种时间区分：发生时间 / 发布时间 / 采集时间 / 有效时间 / 系统记录时间。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone

# 北京时间（+08:00），全系统统一时区，避免跨时区歧义
CST = timezone(timedelta(hours=8))

# uuid5 命名空间：本项目的确定性 ID 空间
NS = uuid.UUID("6f1a2b3c-4d5e-4f60-8a7b-8c9d0e1f2a3b")


def now_iso() -> str:
    """当前北京时间 ISO 字符串。"""
    return datetime.now(CST).isoformat(timespec="milliseconds")


def now_ts() -> float:
    return datetime.now(CST).timestamp()


def iso(dt: datetime) -> str:
    return dt.astimezone(CST).isoformat(timespec="milliseconds")


def det_uuid(*parts) -> str:
    """由业务键确定性生成 UUID（同输入 → 同 ID，重放可复现）。"""
    return str(uuid.uuid5(NS, "\x1f".join(str(p) for p in parts)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 时间解析：把模型输出 / 抓取元信息里的各种日期格式统一成 (lower, upper, precision)
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    ("%Y-%m-%dT%H:%M:%S%z", "second"),
    ("%Y-%m-%d %H:%M:%S", "second"),
    ("%Y-%m-%d %H:%M", "minute"),
    ("%Y-%m-%dT%H:%M:%S", "second"),
    ("%Y-%m-%d", "day"),
    ("%Y.%m.%d", "day"),
    ("%Y年%m月%d日", "day"),
    ("%Y/%m/%d", "day"),
    ("%Y年%m月", "month"),
    ("%Y-%m", "month"),
    ("%Y年", "year"),
    ("%Y", "year"),
]

def parse_datetime(text: str | None):
    """尽力解析日期字符串 → (datetime, precision)；失败返回 (None, None)。"""
    if not text:
        return None, None
    if isinstance(text, (int, float)):
        # Unix 时间戳
        try:
            return datetime.fromtimestamp(float(text), CST), "second"
        except Exception:
            return None, None
    s = str(text).strip().replace("Z", "+00:00")
    # 去掉 "北京时间"、星期等干扰前后缀
    s = re.sub(r"（[^）]*）|\([^)]*\)", "", s).strip()
    for fmt, prec in _DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CST)
            else:
                dt = dt.astimezone(CST)
            return dt, prec
        except ValueError:
            continue
    # 正则兜底：抽取文中第一个形如 2026-09-01 / 2026年9月1日 的日期
    m = re.search(r"(\d{4})[-年./](\d{1,2})[-月./](\d{1,2})[日]?", s)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=CST)
            return dt, "day"
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-年](\d{1,2})", s)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=CST)
            return dt, "month"
        except ValueError:
            pass
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return datetime(int(m.group(0)), 1, 1, tzinfo=CST), "year"
    return None, None


def time_range(precision: str, dt: datetime):
    """按精度把时间点展开为左闭右开区间 [lower, upper)。"""
    spans = {
        "second": timedelta(seconds=1),
        "minute": timedelta(minutes=1),
        "hour": timedelta(hours=1),
        "day": timedelta(days=1),
    }
    if precision in spans:
        return dt, dt + spans[precision]
    if precision == "month":
        if dt.month == 12:
            upper = dt.replace(year=dt.year + 1, month=1)
        else:
            upper = dt.replace(month=dt.month + 1)
        return dt, upper
    if precision == "year":
        return dt, dt.replace(year=dt.year + 1)
    return dt, dt + timedelta(days=1)


_YEAR_MIN, _YEAR_MAX = 2015, 2027  # 语料时代之外的年份视为抽取噪声（如 2006/2088）


def _clamp_year(dt):
    """年份超界 → None（时间未知），否则原样返回。"""
    if dt is not None and (_YEAR_MIN <= dt.year <= _YEAR_MAX):
        return dt
    return None


def normalize_event_time(obj) -> tuple[str | None, str | None, str]:
    """把抽取器输出的 event_time 对象规范化为 (lower, upper, precision) ISO 字符串。"""
    if not obj:
        return None, None, "unknown"
    if isinstance(obj, str):
        dt, prec = parse_datetime(obj)
        dt = _clamp_year(dt)
        if dt is None:
            return None, None, "unknown"
        lo, hi = time_range(prec, dt)
        return iso(lo), iso(hi), prec
    if isinstance(obj, dict):
        prec = obj.get("precision") or "unknown"
        lo_dt, p1 = parse_datetime(obj.get("lower"))
        hi_dt, p2 = parse_datetime(obj.get("upper"))
        if lo_dt is None and hi_dt is None:
            # 尝试 value/point 字段
            dt, p = parse_datetime(obj.get("value") or obj.get("point"))
            dt = _clamp_year(dt)
            if dt is None:
                return None, None, "unknown"
            lo, hi = time_range(p, dt)
            return iso(lo), iso(hi), p
        if lo_dt is not None and hi_dt is not None:
            lo_dt, hi_dt = _clamp_year(lo_dt), _clamp_year(hi_dt)
            if lo_dt is None and hi_dt is None:
                return None, None, "unknown"
            if lo_dt is None:
                return None, iso(hi_dt), prec if prec != "unknown" else (p2 or "day")
            if hi_dt is None:
                lo, _ = time_range(p1 or "day", lo_dt)
                return iso(lo), None, prec if prec != "unknown" else (p1 or "day")
            if hi_dt < lo_dt:
                lo_dt, hi_dt = hi_dt, lo_dt
            return iso(lo_dt), iso(hi_dt), prec if prec != "unknown" else (p1 or p2 or "day")
        if lo_dt is not None:
            lo, _ = time_range(p1 or "day", lo_dt)
            return iso(lo), None, prec if prec != "unknown" else (p1 or "day")
        _, hi = time_range(p2 or "day", hi_dt)
        return None, iso(hi), prec if prec != "unknown" else (p2 or "day")
    return None, None, "unknown"


# ---------------------------------------------------------------------------
# 文本规范化与引文定位（铁律：引用由程序计算字符区间，不信任模型手写 offset）
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_for_match(text: str) -> tuple[str, list[int]]:
    """压缩空白并去掉零宽字符，返回 (规范化文本, 原文索引映射)。"""
    out_chars, mapping = [], []
    for i, ch in enumerate(text):
        if unicodedata.category(ch) in ("Cf", "Zs", "Cc") and ch not in "\n":
            continue
        out_chars.append(ch)
        mapping.append(i)
    s = "".join(out_chars)
    # 再做一次空白压缩（连续空白折叠为单个空格）
    compact, cmap = [], []
    last_space = False
    for j, ch in enumerate(s):
        is_space = ch.isspace()
        if is_space and last_space:
            continue
        last_space = is_space
        compact.append(" " if is_space else ch)
        cmap.append(mapping[j])
    return "".join(compact), cmap


def find_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """在原文中定位引文，返回 (start, end) 字符区间；找不到返回 None。"""
    if not quote or not text:
        return None
    ntext, _ = normalize_for_match(text)
    nquote, _ = normalize_for_match(quote)
    if len(nquote) < 6:
        return None
    pos = ntext.find(nquote)
    if pos >= 0:
        # 规范化文本与原文长度不同，这里返回规范化坐标；调用方保存两份文本
        return pos, pos + len(nquote)
    # 容错：截取引文中间最长 40 字再匹配
    if len(nquote) > 40:
        mid = nquote[len(nquote) // 2 - 20 : len(nquote) // 2 + 20]
        pos = ntext.find(mid)
        if pos >= 0:
            return pos, pos + len(nquote)
    return None


# ---------------------------------------------------------------------------
# JSON 清洗：从模型输出里稳健地取出 JSON 对象
# ---------------------------------------------------------------------------

def extract_json(text: str):
    """从模型回复文本中提取第一个 JSON 对象/数组；失败返回 None。

    包含截断抢救：长输出在 max_tokens 处被切断时，按括号栈补全闭合，
    保住已完整生成的元素（下游仍有 schema 校验兜底）。
    """
    if not text:
        return None
    s = text.strip()
    # 去掉 ```json 代码围栏
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    # 去掉 <think>...</think>（万一 /no_think 失效）
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start < 0:
            continue
        end = s.rfind(closer)
        if end <= start:
            continue
        frag = s[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            # 常见修复：尾逗号
            fixed = re.sub(r",\s*([}\]])", r"\1", frag)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    # 截断抢救：补全未闭合的括号
    start = s.find("{") if "{" in s else s.find("[")
    if start >= 0:
        repaired = _repair_truncated_json(s[start:])
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        # 还不行：逐步剥掉末尾悬挂的键/空容器再试
        frag = s[start:]
        if last_ok := _strip_dangling(frag):
            try:
                return json.loads(last_ok)
            except json.JSONDecodeError:
                pass
    return None


def _strip_dangling(frag: str) -> str | None:
    """循环剥掉末尾不完整的键值/空容器，再补闭合括号。"""
    import re as _re
    frag = _cut_unterminated_string(frag)
    changed = True
    while changed and frag:
        changed = False
        frag2 = _re.sub(r'[,\s]*"[^"]*"\s*:\s*$', "", frag).rstrip()
        if frag2 != frag:
            frag, changed = frag2, True
        frag2 = _re.sub(r'[,\s]*([\[{])\s*$', "", frag).rstrip()
        if frag2 != frag:
            frag, changed = frag2, True
        frag = frag.rstrip().rstrip(",")
    frag = frag.strip()
    if not frag or frag[0] not in "{[":
        return None
    return _repair_truncated_json(frag)


def _cut_unterminated_string(frag: str) -> str:
    """截断发生在字符串值中间时的精确修复。

    找到最后一个闭合的引号 token；若其后原本跟的是冒号（说明它是键、
    值字符串被切断），把 ", 键" 一起删掉；否则它是完整值，保留到它为止。
    """
    instr = esc = False
    intervals = []  # 已闭合字符串 token 的 (open, close)
    open_at = -1
    for i, ch in enumerate(frag):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            if not instr:
                open_at = i
            else:
                intervals.append((open_at, i))
            instr = not instr
    if not instr or not intervals:
        return frag
    _open, close = intervals[-1]
    after = frag[close + 1:].lstrip()
    if after.startswith(":"):
        # 悬挂键：连着前面的逗号一起删除
        head = frag[:_open].rstrip()
        if head.endswith(","):
            head = head[:-1].rstrip()
        return head
    return frag[: close + 1]


def _repair_truncated_json(frag: str) -> str | None:
    """把被截断的 JSON 文本修复到可解析：处理未闭合字符串、悬挂逗号、未闭合括号。"""
    if not frag:
        return None
    frag = _cut_unterminated_string(frag)
    # 统计未闭合括号栈
    stack = []
    instr = esc = False
    for ch in frag:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            instr = not instr
            continue
        if instr:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack:
        return frag
    # 去掉悬挂的尾逗号后补全闭合符
    frag = frag.rstrip()
    frag = frag.rstrip(",")
    return frag + "".join(reversed(stack))


def canonical_json(obj) -> str:
    """稳定 JSON 序列化（键排序），用于哈希与幂等键。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def as_str_list(x) -> list[str]:
    out = []
    for item in as_list(x):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            v = item.get("name") or item.get("value")
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out
