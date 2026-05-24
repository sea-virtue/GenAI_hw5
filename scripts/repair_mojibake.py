from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.encoding_utils import repair_mojibake
from src.thu_qa.io_utils import read_jsonl, write_jsonl
from src.thu_qa.text_utils import clean_text


def repair_file(args: argparse.Namespace) -> None:
    rows = []
    changed = 0
    for row in read_jsonl(args.input):
        fixed = dict(row)
        for field in ("title", "text"):
            old = fixed.get(field, "")
            if not isinstance(old, str):
                continue
            new = repair_mojibake(old)
            if field == "text":
                new = clean_text(new)
            if new != old:
                changed += 1
                fixed[field] = new
        rows.append(fixed)

    write_jsonl(args.output, rows)
    print(f"saved {len(rows)} rows to {args.output}; repaired fields: {changed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-effort repair for UTF-8 pages decoded as latin-1.")
    parser.add_argument("--input", default="data/raw/thu_pages.jsonl")
    parser.add_argument("--output", default="data/raw/thu_pages_repaired.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    repair_file(parse_args())
