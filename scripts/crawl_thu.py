from __future__ import annotations

import argparse
import io
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.io_utils import ensure_parent
from src.thu_qa.text_utils import clean_text, is_useful_url, stable_hash


DEFAULT_ALLOWED_HOST_SUFFIXES = (
    "tsinghua.edu.cn",
    "tsinghua.edu",
)


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
        parser.parse(response.text.splitlines())
        cache[base] = parser
        return parser
    except requests.RequestException:
        cache[base] = None  # type: ignore[assignment]
        return None


def extract_html(url: str, html: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()
    for selector in ["header", "footer", "nav", ".nav", ".footer", ".header", ".breadcrumb"]:
        for tag in soup.select(selector):
            tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)

    main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body or soup
    text = clean_text(main.get_text("\n", strip=True))

    links = []
    for link in soup.find_all("a", href=True):
        next_url = normalize_url(urljoin(url, link["href"]))
        if is_useful_url(next_url):
            links.append(next_url)
    return title.strip(), text, links


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
        title = Path(urlparse(url).path).name or "PDF 文件"
        return "pdf", title, text, []
    response.encoding = response.encoding or "utf-8"
    title, text, links = extract_html(url, response.text)
    return "html", title, text, links


def crawl(args: argparse.Namespace) -> None:
    seeds = [normalize_url(url) for url in load_seed_urls(Path(args.seeds))]
    allowed_suffixes = tuple(args.allowed_host_suffix)
    queue = deque((url, 0) for url in seeds if allowed_host(url, allowed_suffixes))
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
        if not allowed_host(url, allowed_suffixes):
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

        text_key = stable_hash(text[:5000])
        if len(text) >= args.min_chars and text_key not in text_hashes:
            text_hashes.add(text_key)
            rows.append(
                {
                    "url": url,
                    "title": title,
                    "source_type": source_type,
                    "text": text,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            progress.update(1)

        if depth < args.max_depth:
            for link in links:
                link = normalize_url(link)
                if link not in visited and allowed_host(link, allowed_suffixes):
                    queue.append((link, depth + 1))
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
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--allowed-host-suffix", action="append", default=list(DEFAULT_ALLOWED_HOST_SUFFIXES))
    parser.add_argument("--user-agent", default="THU-GenAI-HW5-CampusQA/1.0")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())
