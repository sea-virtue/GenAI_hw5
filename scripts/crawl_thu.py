from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.encoding_utils import decode_html_content, looks_mojibake, repair_mojibake
from src.thu_qa.io_utils import ensure_parent
from src.thu_qa.text_utils import clean_text, is_useful_url, stable_hash


DEFAULT_ALLOWED_HOST_SUFFIXES = (
    "tsinghua.edu.cn",
    "tsinghua.edu",
)
DEFAULT_EXCLUDE_URL_KEYWORDS = (
    "news.tsinghua.edu.cn",
    "/news/",
    "/xw",
    "xwdt",
    "tzgg",
    "notice",
    "media",
)
BAD_CONTENT_KEYWORDS = (
    "\u65b0\u95fb",
    "\u52a8\u6001",
    "\u901a\u77e5\u516c\u544a",
    "\u5b66\u672f\u6d3b\u52a8",
    "\u8bb2\u5ea7",
    "\u66f4\u591a\u65b0\u95fb",
    "NEWS",
    "Notices",
)
SERVICE_CONTENT_KEYWORDS = (
    "\u670d\u52a1",
    "\u6307\u5357",
    "\u5bfc\u5f15",
    "\u89c4\u5219",
    "\u529e\u7406",
    "\u9884\u7ea6",
    "\u501f\u9605",
    "\u501f\u8fd8",
    "\u5f00\u9986",
    "\u65f6\u95f4",
    "\u6821\u5386",
    "\u4ea4\u901a",
    "\u5730\u56fe",
    "\u53c2\u89c2",
    "\u8054\u7cfb",
    "\u54a8\u8be2",
    "\u62db\u751f",
    "\u5c31\u4e1a",
    "\u57f9\u517b",
    "\u6559\u5b66",
    "\u79d1\u7814",
    "\u6982\u51b5",
    "\u4ecb\u7ecd",
    "\u4fe1\u606f\u516c\u5f00",
)
DISCOVERY_ONLY_TITLE_KEYWORDS = (
    "\u5e08\u8d44",
    "\u901a\u77e5\u516c\u544a",
    "\u65b0\u95fb\u52a8\u6001",
    "\u66f4\u591a",
)
DEFAULT_FOCUS_URLS = (
    "https://www.tsinghua.edu.cn/yxsz.htm",
)
MAIN_HOSTS = {
    "www.tsinghua.edu.cn",
    "tsinghua.edu.cn",
}
STABLE_INFO_KEYWORDS = (
    "about",
    "intro",
    "overview",
    "profile",
    "faculty",
    "research",
    "education",
    "teaching",
    "program",
    "admission",
    "contact",
    "gk",
    "jj",
    "xygk",
    "yxgk",
    "xxgk",
    "xyjj",
    "xkjs",
    "kxyj",
    "jx",
    "jyjx",
    "rcpy",
    "sz",
    "szdw",
    "lxwm",
    "service",
    "guide",
    "rule",
    "calendar",
    "traffic",
    "map",
    "library",
    "lib",
    "kgsj",
    "jydy",
    "jhfw",
    "qhxl",
    "syxx",
)

CONTENT_SELECTORS = [
    "article",
    "main",
    ".v_news_content",
    "#vsb_content",
    "#vsb_content_2",
    ".wp_articlecontent",
    ".article-content",
    ".article_content",
    ".news_content",
    ".main-content",
    ".detail",
    ".detail-content",
    ".content",
    ".con",
    ".text",
    ".txt",
]

DROP_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "header",
    "footer",
    "nav",
    "aside",
    ".nav",
    ".navbar",
    ".menu",
    ".footer",
    ".header",
    ".breadcrumb",
    ".crumb",
    ".sidebar",
    ".subnav",
    ".top",
    ".bottom",
    ".links",
    ".pagination",
]

BOILERPLATE_LINE_PATTERNS = [
    "English",
    "\u6e05\u534e\u4e3b\u9875",
    "\u6e05\u534e\u65b0\u95fb",
    "\u8bbf\u5ba2",
    "\u6821\u53cb",
    "\u6350\u8d60",
    "\u4eba\u624d\u62db\u8058",
    "\u5185\u7f51",
    "\u90ae\u7bb1",
    "\u56fe\u4e66\u9986",
    "\u9996\u9875",
    "\u5bfc\u822a\u83dc\u5355",
    "\u4e0a\u4e00\u9875",
    "\u4e0b\u4e00\u9875",
    "\u5c3e\u9875",
    "\u8df3\u8f6c",
    "\u5e38\u7528\u94fe\u63a5",
    "\u610f\u89c1\u53cd\u9988",
    "\u7248\u6743\u6240\u6709",
    "Copyright",
    "ICP",
    "webmaster@",
]


def load_seed_urls(path: Path) -> list[str]:
    urls = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean.rstrip("/")


def allowed_host(url: str, suffixes: Iterable[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in suffixes)


def allowed_url(url: str, suffixes: Iterable[str], excluded_keywords: Iterable[str]) -> bool:
    if not allowed_host(url, suffixes):
        return False
    lowered = url.lower()
    return not any(keyword.lower() in lowered for keyword in excluded_keywords)


def is_focus_url(url: str, focus_urls: Iterable[str]) -> bool:
    normalized = normalize_url(url)
    return any(normalized == normalize_url(focus_url) for focus_url in focus_urls)


def is_department_like_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "yxsz" in path:
        return True
    if host.endswith(".tsinghua.edu.cn") and host not in MAIN_HOSTS:
        ignored_hosts = {
            "info.tsinghua.edu.cn",
            "mails.tsinghua.edu.cn",
            "jobs.tsinghua.edu.cn",
            "news.tsinghua.edu.cn",
        }
        return host not in ignored_hosts
    return False


def is_root_or_index_url(url: str) -> bool:
    path = urlparse(url).path.lower().strip("/")
    return path in ("", "index.htm", "index.html", "index.jsp", "main.htm", "main.html")


def has_stable_info_signal(url: str, title: str = "", text: str = "") -> bool:
    lowered = url.lower()
    combined = title + "\n" + text[:500]
    if any(keyword in lowered for keyword in STABLE_INFO_KEYWORDS):
        return True
    return any(keyword in combined for keyword in SERVICE_CONTENT_KEYWORDS)


def link_priority(url: str, focus_urls: Iterable[str]) -> tuple[int, int, str]:
    if is_focus_url(url, focus_urls):
        return (0, len(url), url)
    lowered = url.lower()
    if is_department_like_url(url):
        if any(keyword in lowered for keyword in STABLE_INFO_KEYWORDS):
            return (1, len(url), url)
        parsed = urlparse(url)
        if parsed.path in ("", "/") or parsed.path.endswith(("index.htm", "index.html")):
            return (2, len(url), url)
        return (3, len(url), url)
    host = urlparse(url).netloc.lower()
    if host in MAIN_HOSTS:
        return (5, len(url), url)
    return (4, len(url), url)


def should_enqueue_link(current_url: str, link: str, depth: int, args: argparse.Namespace) -> bool:
    if args.department_crawl and is_focus_url(current_url, args.focus_url):
        return is_department_like_url(link) or is_focus_url(link, args.focus_url)
    if args.department_crawl and depth >= 1:
        current_host = urlparse(current_url).netloc.lower()
        link_host = urlparse(link).netloc.lower()
        if current_host not in MAIN_HOSTS and link_host != current_host:
            return False
    return True


def get_robot_parser(session: requests.Session, url: str, cache: dict[str, RobotFileParser]) -> RobotFileParser | None:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base in cache:
        return cache[base]
    robots_url = urljoin(base, "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = session.get(robots_url, timeout=10)
        if response.status_code >= 400:
            cache[base] = None  # type: ignore[assignment]
            return None
        html = decode_html_content(response.content, response.headers.get("content-type", ""), response.apparent_encoding)
        parser.parse(html.splitlines())
        cache[base] = parser
        return parser
    except requests.RequestException:
        cache[base] = None  # type: ignore[assignment]
        return None


def extract_links(url: str, soup: BeautifulSoup) -> list[str]:
    links = []
    for link in soup.find_all("a", href=True):
        next_url = normalize_url(urljoin(url, link["href"]))
        if is_useful_url(next_url):
            links.append(next_url)
    return links


def prune_noise(soup: BeautifulSoup) -> None:
    for selector in DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()


def clean_page_lines(text: str) -> str:
    lines = []
    for raw in clean_text(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(pattern in line for pattern in BOILERPLATE_LINE_PATTERNS):
            continue
        if len(line) <= 4 and line.isdigit():
            continue
        lines.append(line)
    for idx, line in enumerate(lines):
        has_sentence_mark = any(ch in line for ch in "\u3002\uff1b\uff1a\uff0c")
        if len(line) >= 30 and has_sentence_mark:
            lines = lines[idx:]
            break
    return clean_text("\n".join(lines))


def node_score(node) -> float:
    text = clean_page_lines(node.get_text("\n", strip=True))
    if len(text) < 80:
        return -1.0
    link_text_len = sum(len(a.get_text("", strip=True)) for a in node.find_all("a"))
    link_ratio = link_text_len / max(len(text), 1)
    punctuation = sum(text.count(ch) for ch in "\u3002\uff1b\uff1a\uff0c")
    paragraph_count = len(node.find_all(["p", "li", "section"]))
    boilerplate_hits = sum(text.count(pattern) for pattern in BOILERPLATE_LINE_PATTERNS)
    return len(text) + punctuation * 20 + paragraph_count * 8 - link_ratio * len(text) * 1.8 - boilerplate_hits * 120


def select_content_node(soup: BeautifulSoup):
    candidates = []
    for selector in CONTENT_SELECTORS:
        candidates.extend(soup.select(selector))
    candidates.extend(soup.find_all(["article", "main", "section", "div"]))
    if not candidates:
        return soup.body or soup
    return max(candidates, key=node_score)


def extract_title(soup: BeautifulSoup) -> str:
    candidates = []
    title_tag = soup.find("title")
    if title_tag:
        candidates.append((clean_text(title_tag.get_text(" ", strip=True)), 30))
    for selector, weight in [
        ("h1", 25),
        (".article-title", 20),
        (".arti_title", 20),
        (".title", 10),
        ("h2", 5),
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get_text(strip=True):
            candidates.append((clean_text(tag.get_text(" ", strip=True)), weight))

    def score(candidate: tuple[str, int]) -> float:
        title, weight = candidate
        if not title:
            return -1
        boilerplate_hits = sum(pattern in title for pattern in BOILERPLATE_LINE_PATTERNS)
        return weight + min(len(title), 60) * 0.2 - boilerplate_hits * 100

    return max(candidates, key=score)[0] if candidates else ""


def is_quality_page(title: str, text: str, min_chars: int) -> bool:
    return is_saveable_page("", title, text, min_chars, "html")


def text_shape_metrics(text: str) -> dict[str, float]:
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    if not lines:
        return {
            "line_count": 0,
            "short_line_ratio": 1.0,
            "sentence_line_ratio": 0.0,
            "date_line_ratio": 0.0,
        }
    sentence_marks = "\u3002\uff1b\uff1a\uff0c\uff1f\uff01"
    sentence_lines = [line for line in lines if len(line) >= 20 and any(mark in line for mark in sentence_marks)]
    short_lines = [line for line in lines if len(line) <= 12]
    date_lines = [line for line in lines if re.search(r"(\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}|[A-Z][a-z]{2}\s+\d{1,2})", line)]
    return {
        "line_count": len(lines),
        "short_line_ratio": len(short_lines) / len(lines),
        "sentence_line_ratio": len(sentence_lines) / len(lines),
        "date_line_ratio": len(date_lines) / len(lines),
    }


def is_saveable_page(url: str, title: str, text: str, min_chars: int, source_type: str) -> bool:
    stable_signal = has_stable_info_signal(url, title, text)
    stable_url_title_signal = has_stable_info_signal(url, title, "")
    min_required_chars = 80 if stable_signal else min_chars
    if len(text) < min_required_chars:
        return False
    if looks_mojibake(title) or looks_mojibake(text):
        return False
    chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    if chinese_chars / max(len(text), 1) < 0.25:
        return False
    boilerplate_hits = sum(text.count(pattern) for pattern in BOILERPLATE_LINE_PATTERNS)
    if boilerplate_hits >= 5:
        return False
    if source_type == "pdf":
        return True

    if is_department_like_url(url) and is_root_or_index_url(url):
        return False

    bad_signal = any(keyword in title or keyword in text[:300] for keyword in BAD_CONTENT_KEYWORDS)
    if bad_signal and not stable_url_title_signal:
        return False

    first_line = next((line.strip() for line in clean_text(text).splitlines() if line.strip()), "")
    starts_like_news = bool(
        re.match(r"^(\d{4}\u5e74)?\d{1,2}\u6708\d{1,2}\u65e5", first_line)
        or re.match(r"^[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}", first_line)
    )
    if starts_like_news and not stable_url_title_signal:
        return False

    if any(keyword == title.strip() for keyword in DISCOVERY_ONLY_TITLE_KEYWORDS):
        return False

    metrics = text_shape_metrics(text)
    if (
        metrics["line_count"] >= 18
        and metrics["short_line_ratio"] > 0.65
        and metrics["sentence_line_ratio"] < 0.18
        and not stable_url_title_signal
    ):
        return False
    if metrics["line_count"] >= 12 and metrics["date_line_ratio"] > 0.25 and not stable_url_title_signal:
        return False
    if metrics["sentence_line_ratio"] < 0.08 and not stable_signal:
        return False
    return True


def is_recordable_page(title: str, text: str, min_chars: int) -> bool:
    if len(text) < max(60, min_chars // 2):
        return False
    if looks_mojibake(title) or looks_mojibake(text):
        return False
    chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return chinese_chars / max(len(text), 1) >= 0.18


def classify_page(url: str, title: str, text: str, min_chars: int, source_type: str) -> tuple[bool, str]:
    qa_candidate = is_saveable_page(url, title, text, min_chars, source_type)
    if qa_candidate:
        return True, "qa_candidate"
    metrics = text_shape_metrics(text)
    if is_department_like_url(url) and is_root_or_index_url(url):
        return False, "department_homepage_discovery"
    if metrics["line_count"] >= 15 and metrics["short_line_ratio"] > 0.7:
        return False, "list_or_table_discovery"
    if any(keyword in title or keyword in text[:300] for keyword in BAD_CONTENT_KEYWORDS):
        return False, "news_or_notice_discovery"
    return False, "discovery_or_low_signal"


def extract_html(url: str, html: str) -> tuple[str, str, list[str]]:
    html = repair_mojibake(html)
    soup = BeautifulSoup(html, "lxml")
    links = extract_links(url, soup)
    title = extract_title(soup)
    prune_noise(soup)
    main = select_content_node(soup)
    text = clean_page_lines(main.get_text("\n", strip=True))
    return repair_mojibake(title).strip(), repair_mojibake(text), links


def extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return clean_text("\n".join(pages))


def fetch_page(session: requests.Session, url: str) -> tuple[str, str, str, list[str]]:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = extract_pdf(response.content)
        title = Path(urlparse(url).path).name or "PDF \u6587\u4ef6"
        return "pdf", title, text, []
    html = decode_html_content(response.content, content_type, response.apparent_encoding)
    title, text, links = extract_html(url, html)
    return "html", title, text, links


def crawl(args: argparse.Namespace) -> None:
    seeds = [normalize_url(url) for url in load_seed_urls(Path(args.seeds))]
    for focus_url in args.focus_url:
        normalized_focus = normalize_url(focus_url)
        if normalized_focus not in seeds:
            seeds.insert(0, normalized_focus)
    seeds = sorted(seeds, key=lambda url: link_priority(url, args.focus_url))
    allowed_suffixes = tuple(args.allowed_host_suffix)
    excluded_keywords = tuple(args.exclude_url_keyword)
    queue = deque((url, 0) for url in seeds if allowed_url(url, allowed_suffixes, excluded_keywords))
    visited = set()
    text_hashes = set()
    robot_cache: dict[str, RobotFileParser] = {}
    rows = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        }
    )

    progress = tqdm(total=args.max_pages, desc="crawl")
    while queue and len(rows) < args.max_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not allowed_url(url, allowed_suffixes, excluded_keywords):
            continue

        robot = get_robot_parser(session, url, robot_cache)
        if robot is not None and not robot.can_fetch(args.user_agent, url):
            continue

        try:
            source_type, title, text, links = fetch_page(session, url)
        except Exception as exc:
            if args.verbose:
                print(f"[skip] {url}: {exc}")
            continue

        qa_candidate, page_role = classify_page(url, title, text, args.min_chars, source_type)
        recordable = qa_candidate or (args.save_discovery_pages and is_recordable_page(title, text, args.min_chars))
        text_key = stable_hash(text[:5000])
        if recordable and text_key not in text_hashes:
            text_hashes.add(text_key)
            rows.append(
                {
                    "url": url,
                    "title": title,
                    "source_type": source_type,
                    "text": text,
                    "qa_candidate": qa_candidate,
                    "page_role": page_role,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            progress.update(1)

        if depth < args.max_depth:
            next_items = []
            sorted_links = sorted(set(links), key=lambda item: link_priority(item, args.focus_url))
            for link in sorted_links:
                link = normalize_url(link)
                if (
                    link not in visited
                    and allowed_url(link, allowed_suffixes, excluded_keywords)
                    and should_enqueue_link(url, link, depth, args)
                ):
                    next_items.append((link, depth + 1))
            if is_focus_url(url, args.focus_url):
                for item in reversed(next_items):
                    queue.appendleft(item)
            else:
                queue.extend(next_items)
        time.sleep(args.delay)

    progress.close()
    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} pages to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl public Tsinghua web pages for campus QA data.")
    parser.add_argument("--seeds", default="configs/crawl_seed_urls.txt")
    parser.add_argument("--output", default="data/raw/thu_pages.jsonl")
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--allowed-host-suffix", action="append", default=list(DEFAULT_ALLOWED_HOST_SUFFIXES))
    parser.add_argument("--exclude-url-keyword", action="append", default=list(DEFAULT_EXCLUDE_URL_KEYWORDS))
    parser.add_argument("--focus-url", action="append", default=list(DEFAULT_FOCUS_URLS))
    parser.add_argument(
        "--save-discovery-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save decoded but low-signal discovery/list pages in raw JSONL with qa_candidate=false.",
    )
    parser.add_argument(
        "--department-crawl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize school/department links from yxsz.htm and stay within department hosts after depth 1.",
    )
    parser.add_argument("--user-agent", default="THU-GenAI-HW5-CampusQA/1.0")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())
