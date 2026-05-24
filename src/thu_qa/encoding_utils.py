from __future__ import annotations

import re
from html import unescape


_META_CHARSET_RE = re.compile(br"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9._-]+)", re.I)
_HTTP_CHARSET_RE = re.compile(r"charset=([A-Za-z0-9._-]+)", re.I)
_C1_RE = re.compile(r"[\u0080-\u009f]")
_LATIN1_MARKERS = (
    "\u00c3",
    "\u00c2",
    "\u00e2",
    "\u00e5",
    "\u00e6",
    "\u00e7",
    "\u00e8",
    "\u00e9",
    "\u00ef",
)
_COMMON_GBK_GARBAGE = (
    "\u93c2",
    "\u626e",
    "\u73db",
    "\u95b2",
    "\u9358",
    "\u5a13",
    "\u546d",
    "\u5d15",
    "\u6f76",
    "\u6fbe",
    "\u6fcb",
    "\u934f",
    "\u93ac",
    "\u9416",
    "\u93ad",
    "\u95c1",
    "\u9547",
    "\u951b",
    "\u9518",
)


def _unique(items: list[str | None]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if not item:
            continue
        normalized = item.strip().strip("\"'").lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_declared_charsets(content: bytes, content_type: str = "") -> list[str]:
    candidates: list[str | None] = []
    http_match = _HTTP_CHARSET_RE.search(content_type or "")
    if http_match:
        candidates.append(http_match.group(1))

    head = content[:4096]
    meta_match = _META_CHARSET_RE.search(head)
    if meta_match:
        candidates.append(meta_match.group(1).decode("ascii", errors="ignore"))
    return _unique(candidates)


def _count_cjk(text: str) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text)


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    c1_count = len(_C1_RE.findall(text))
    private_count = sum("\ue000" <= ch <= "\uf8ff" for ch in text)
    latin_count = sum(text.count(marker) for marker in _LATIN1_MARKERS)
    gbk_count = sum(text.count(marker) for marker in _COMMON_GBK_GARBAGE)
    replacement_count = text.count("\ufffd")
    return c1_count >= 2 or private_count >= 2 or latin_count >= 8 or gbk_count >= 3 or replacement_count >= 2


def _decode_score(text: str) -> float:
    chinese = _count_cjk(text)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    replacement = text.count("\ufffd")
    c1 = len(_C1_RE.findall(text))
    latin_marker = sum(text.count(marker) for marker in _LATIN1_MARKERS)
    gbk_marker = sum(text.count(marker) for marker in _COMMON_GBK_GARBAGE)
    punctuation = sum(text.count(ch) for ch in "\u3002\uff0c\uff1a\uff1b\uff08\uff09")
    return (
        chinese * 3
        + ascii_letters * 0.02
        + punctuation * 0.5
        - replacement * 30
        - c1 * 10
        - latin_marker * 3
        - gbk_marker * 8
    )


def decode_html_content(content: bytes, content_type: str = "", apparent_encoding: str | None = None) -> str:
    candidates = extract_declared_charsets(content, content_type)
    # Prefer UTF-8 for modern Tsinghua pages; many servers omit charset and requests may guess GBK.
    candidates += ["utf-8-sig", "utf-8", "gb18030", apparent_encoding, "big5", "latin-1"]
    decoded = []
    for encoding in _unique(candidates):
        try:
            text = content.decode(encoding, errors="replace")
        except LookupError:
            continue
        decoded.append((encoding, text, _decode_score(text)))
    if not decoded:
        return content.decode("utf-8", errors="replace")
    decoded.sort(key=lambda item: item[2], reverse=True)
    return decoded[0][1]


def _repair_with_encoding(text: str, source_encoding: str) -> str | None:
    try:
        return text.encode(source_encoding, errors="ignore").decode("utf-8", errors="replace")
    except UnicodeError:
        return None


def repair_mojibake(text: str) -> str:
    if not text:
        return text

    text = unescape(text).replace("\ufeff", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = text.replace("\u9518\ufffd", "").replace("\ufffd", "")
    compact_latin = re.sub(r"(?<=[\u0080-\u00ff])\n(?=[\u0080-\u00ff])", "", text)
    compact_gbk = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    candidates = [text, compact_latin, compact_gbk]

    for source_encoding in ("latin-1", "gb18030"):
        for candidate in (text, compact_latin, compact_gbk):
            fixed = _repair_with_encoding(candidate, source_encoding)
            if fixed and (_count_cjk(fixed) >= 2 or len(fixed.strip()) >= max(6, len(candidate.strip()) * 0.25)):
                candidates.append(unescape(fixed).replace("\ufeff", ""))

    best = max(candidates, key=_decode_score)
    if _decode_score(best) > _decode_score(text) + 10:
        return best
    return text


def repair_latin1_mojibake(text: str) -> str:
    return repair_mojibake(text)
