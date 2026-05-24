from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.io_utils import read_jsonl, write_jsonl
from src.thu_qa.text_utils import stable_hash


EVAL_PROMPT_PREFIXES = [
    "请回答：",
    "我想确认一下：",
    "根据清华大学公开信息，",
    "关于清华大学相关信息，",
    "能不能说明一下：",
]


def normalize_manual_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen = set()
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue
        key = stable_hash(instruction + "\n" + output)
        if key in seen:
            continue
        seen.add(key)

        item = {
            "instruction": instruction,
            "input": str(row.get("input", "")).strip(),
            "output": output,
        }
        for meta_key in ["source_file", "source_line", "source_title", "source_url", "data_source"]:
            if meta_key in row:
                item[meta_key] = row[meta_key]
        normalized.append(item)
    return normalized


def add_ids(rows: list[dict], prefix: str = "thuqa") -> list[dict]:
    result = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["id"] = f"{prefix}-{idx:05d}"
        result.append(item)
    return result


def convert_sharegpt(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        user_content = row["instruction"]
        if row.get("input"):
            user_content += "\n" + row["input"]
        converted.append(
            {
                "id": row["id"],
                "conversations": [
                    {"from": "human", "value": user_content},
                    {"from": "gpt", "value": row["output"]},
                ],
                "source_file": row.get("source_file", ""),
                "source_line": row.get("source_line", ""),
                "source_url": row.get("source_url", ""),
                "source_title": row.get("source_title", ""),
            }
        )
    return converted


def select_eval_sources(rows: list[dict], eval_size: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("source_file") or row.get("source_title") or "manual")
        groups.setdefault(group, []).append(row)

    rng = random.Random(seed)
    group_names = sorted(groups)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    rng.shuffle(group_names)

    selected: list[dict] = []
    while len(selected) < eval_size and group_names:
        progressed = False
        for group in list(group_names):
            if groups[group]:
                selected.append(groups[group].pop())
                progressed = True
                if len(selected) >= eval_size:
                    break
            else:
                group_names.remove(group)
        if not progressed:
            break
    return selected


def paraphrase_for_eval(row: dict, idx: int) -> dict:
    item = dict(row)
    item["id"] = f"thuqa-eval-{idx:05d}"
    item["instruction"] = EVAL_PROMPT_PREFIXES[idx % len(EVAL_PROMPT_PREFIXES)] + row["instruction"]
    item["eval_source_id"] = row["id"]
    item["split_strategy"] = "paraphrased_question_seen_fact"
    return item


def split_manual_qa(
    examples: list[dict],
    eval_size: int,
    seed: int,
    split_mode: str,
) -> tuple[list[dict], list[dict]]:
    selected = select_eval_sources(examples, eval_size, seed)
    if split_mode == "holdout":
        eval_ids = {row["id"] for row in selected}
        train_rows = [row for row in examples if row["id"] not in eval_ids]
        eval_rows = [dict(row, split_strategy="exact_holdout") for row in selected]
        return train_rows, eval_rows

    eval_rows = [paraphrase_for_eval(row, idx) for idx, row in enumerate(selected)]
    train_rows = list(examples)
    return train_rows, eval_rows


def build_dataset(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError(f"empty input file: {args.input}")
    if not all("instruction" in row and "output" in row for row in rows):
        raise ValueError(
            "build_dataset.py expects a curated QA JSONL file with instruction/output fields. "
            "Use data/processed/thu_qa_manual_all.jsonl."
        )

    examples = add_ids(normalize_manual_rows(rows), prefix="thuqa")
    if len(examples) < args.min_total:
        print(f"warning: only {len(examples)} QA examples; target min_total is {args.min_total}.")

    eval_size = min(args.eval_size, max(1, len(examples) - 1))
    train_rows, eval_rows = split_manual_qa(examples, eval_size, args.seed, args.split_mode)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.all_output, examples)
    write_jsonl(args.sharegpt_output, convert_sharegpt(train_rows))

    print(f"input: {args.input}")
    print(f"all:   {len(examples)} -> {args.all_output}")
    print(f"train: {len(train_rows)} -> {args.train_output}")
    print(f"eval:  {len(eval_rows)} -> {args.eval_output}")
    print(f"split_mode: {args.split_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/eval files from manually curated THU QA JSONL.")
    parser.add_argument("--input", default="data/processed/thu_qa_manual_all.jsonl")
    parser.add_argument("--train-output", default="data/processed/train.jsonl")
    parser.add_argument("--eval-output", default="data/processed/eval.jsonl")
    parser.add_argument("--all-output", default="data/processed/qa_all.jsonl")
    parser.add_argument("--sharegpt-output", default="data/processed/train_sharegpt.jsonl")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--min-total", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["qa-paraphrase", "holdout"],
        default="qa-paraphrase",
        help=(
            "qa-paraphrase keeps all canonical facts in train and evaluates on paraphrased questions. "
            "holdout removes exact eval rows from train, but facts may be unseen."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
