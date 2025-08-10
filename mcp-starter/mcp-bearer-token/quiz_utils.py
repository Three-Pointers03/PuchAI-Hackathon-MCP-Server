from __future__ import annotations

import re
from typing import Any


VALID_AXES = {"EI", "SN", "TF", "JP"}


def normalize_axis(axis: str | None) -> str | None:
    if not axis:
        return None
    a = str(axis).strip().upper()
    return a if a in VALID_AXES else None


def clamp_value(v: float, low: float = -3.0, high: float = 3.0) -> float:
    return max(low, min(high, v))


def _letter_to_value(letter: str) -> float | None:
    if not letter:
        return None
    ch = str(letter).strip().lower()
    map7 = {"a": -3, "b": -2, "c": -1, "d": 0, "e": 1, "f": 2, "g": 3}
    if ch in map7:
        return float(map7[ch])
    # Also support 4-choice mapping if letters limited to a-d
    map4 = {"a": -3, "b": -1, "c": 1, "d": 3}
    if ch in map4:
        return float(map4[ch])
    return None


def _text_to_value(text: str) -> float | None:
    if not text:
        return None
    s = str(text).strip().lower()
    # Attempt numeric first
    try:
        return float(s)
    except Exception:
        pass

    # Likert text mapping (robust to extra words)
    # Order matters: check for "strongly" variants before generic ones
    if "strong" in s and "agree" in s:
        return 3.0
    if "slight" in s and "agree" in s:
        return 1.0
    if "agree" in s:
        return 2.0
    if "neutral" in s or "neither" in s:
        return 0.0
    if "slight" in s and "disagree" in s:
        return -1.0
    if "strong" in s and "disagree" in s:
        return -3.0
    if "disagree" in s:
        return -2.0

    # Abbreviations
    abbr = {
        "sa": 3.0,
        "a": 2.0,  # note: bare 'a' is ambiguous; keep for text-only paths
        "n": 0.0,
        "sd": -3.0,
        "d": -2.0,
    }
    if s in abbr:
        return float(abbr[s])
    return None


def sanitize_value(value: Any) -> float | None:
    # Numeric directly
    if isinstance(value, (int, float)):
        return clamp_value(float(value))
    # Attempt to interpret strings
    if isinstance(value, str):
        # Try letter mapping first
        v = _letter_to_value(value)
        if v is not None:
            return v
        v = _text_to_value(value)
        if v is not None:
            return clamp_value(v)
        # Try to find trailing/embedded letter token
        m = re.search(r"\b([a-gA-G])\b", value)
        if m:
            v = _letter_to_value(m.group(1))
            if v is not None:
                return v
        # Try numeric substring
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return clamp_value(float(m.group(0)))
            except Exception:
                return None
    return None


def sanitize_responses(responses: list[dict[str, Any]] | list[Any]) -> list[dict[str, float]]:
    sanitized: list[dict[str, float]] = []
    if not isinstance(responses, list):
        return sanitized

    for item in responses:
        axis: str | None = None
        value_obj: Any = None

        if isinstance(item, dict):
            # Accept varied casing and minor key differences
            key_map = {str(k).strip().lower(): k for k in item.keys()}
            axis_key = key_map.get("axis")
            value_key = key_map.get("value")
            if axis_key is not None:
                axis = normalize_axis(item.get(axis_key))
            if value_key is not None:
                value_obj = item.get(value_key)
            # Fallbacks: support schemas like {id:'EI-1', answer:'a'}
            if axis is None and "id" in key_map:
                possible_axis = str(item.get(key_map["id"]))
                m = re.match(r"\s*([A-Za-z]{2})\s*[-:_ ]?", possible_axis or "")
                if m:
                    axis = normalize_axis(m.group(1))
            if value_obj is None:
                for alias in ("answer", "ans", "choice"):
                    alias_key = key_map.get(alias)
                    if alias_key is not None:
                        value_obj = item.get(alias_key)
                        break
        elif isinstance(item, str):
            # Accept lines like 'EI: a', 'sn=-1', 'tf   strongly agree'
            m = re.search(
                r"(?i)\b(EI|SN|TF|JP)\b\s*[:=\-]?\s*(.+)$",
                item.strip(),
            )
            if m:
                axis = normalize_axis(m.group(1))
                value_obj = m.group(2)
        else:
            # Unsupported item type
            continue

        if not axis:
            continue

        val = sanitize_value(value_obj)
        if val is None:
            continue
        sanitized.append({"axis": axis, "value": float(val)})

    return sanitized


def parse_compact_tokens(text: str) -> list[tuple[int, str]]:
    """Return list of (index, raw_value) pairs.

    Supports:
    - 1a 2b 3c
    - 1:a 2=b 3 - c
    - 1:-2 2 1 3 3.0
    - 1 strongly agree, 2 neutral, 3 slightly disagree
    """
    if not text:
        return []
    out: list[tuple[int, str]] = []
    seen: set[int] = set()

    patterns = [
        r"(\d+)\s*[:=\-]?\s*([a-gA-G])",  # letter form
        r"(\d+)\s*[:=\-]?\s*(-?\d+(?:\.\d+)?)",  # numeric form
        r"(?i)(\d+)\s*[:=\-]?\s*((?:strong(?:ly)?\s+)?(?:dis)?agree|slight(?:ly)?\s+(?:dis)?agree|neutral|neither[\w\s]*)",
    ]
    for pat in patterns:
        for idx_str, val in re.findall(pat, text):
            try:
                idx = int(idx_str)
            except Exception:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            out.append((idx, val))
    return out


