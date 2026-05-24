from __future__ import annotations

import hashlib
import re
import string
from typing import Iterable


_SPACE_RE = re.compile(r"\s+")
_URL_DROP_RE = re.compile(r"(javascript:|mailto:|tel:)", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f!?\uff1b;])")
_PUNCT_TABLE = str.maketrans(
    "",
    "",
    string.punctuation
    + "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff08\uff09\u3010\u3011"
    + "\u300a\u300b\u201c\u201d\u2018\u2019\u2014\u2026\u00b7",
)


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060]", "", text)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = _SPACE_RE.sub(" ", raw).strip()
        if len(line) < 2:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines).strip()


def is_useful_url(url: str) -> bool:
    if _URL_DROP_RE.search(url):
        return False
    lowered = url.lower()
    blocked_suffixes = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".zip",
        ".rar",
        ".7z",
        ".mp4",
        ".mp3",
        ".css",
        ".js",
    )
    return not lowered.endswith(blocked_suffixes)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    sentences = []
    for part in parts:
        sentence = _SPACE_RE.sub(" ", part).strip()
        if 12 <= len(sentence) <= 260:
            sentences.append(sentence)
    return sentences


def chunk_text(text: str, max_chars: int = 520, overlap: int = 80) -> list[str]:
    compact = _SPACE_RE.sub(" ", text).strip()
    if not compact:
        return []
    chunks = []
    start = 0
    while start < len(compact):
        end = min(start + max_chars, len(compact))
        cut = compact.rfind("\u3002", start, end)
        if cut > start + max_chars // 2:
            end = cut + 1
        chunk = compact[start:end].strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
        if end >= len(compact):
            break
        start = max(0, end - overlap)
    return chunks


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = _SPACE_RE.sub("", text)
    return text.translate(_PUNCT_TABLE)


def char_f1(prediction: str, reference: str) -> float:
    pred = normalize_for_match(prediction)
    ref = normalize_for_match(reference)
    if not pred or not ref:
        return 0.0
    pred_chars = list(pred)
    ref_chars = list(ref)
    ref_counts = {}
    for ch in ref_chars:
        ref_counts[ch] = ref_counts.get(ch, 0) + 1
    common = 0
    for ch in pred_chars:
        if ref_counts.get(ch, 0) > 0:
            common += 1
            ref_counts[ch] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for text in texts:
        key = normalize_for_match(text)[:220]
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result
