from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.io_utils import read_jsonl, write_jsonl
from src.thu_qa.text_utils import chunk_text, clean_text, split_sentences, stable_hash


QUESTION_TEMPLATES = [
    "请概括介绍{title}。",
    "关于{title}，有哪些重要信息？",
    "如果同学想了解{title}，应该重点注意什么？",
]

KEYWORD_QUESTIONS = [
    (re.compile(r"(地址|位于|地点|校区|路线|交通)"), "关于{title}的地点或位置，有哪些信息？"),
    (re.compile(r"(时间|开放|闭馆|服务时间|截止|安排)"), "关于{title}的时间安排，有哪些信息？"),
    (re.compile(r"(电话|邮箱|联系|咨询|网址|平台)"), "如何联系或咨询{title}相关事务？"),
    (re.compile(r"(申请|办理|预约|注册|提交|流程|步骤)"), "办理{title}相关事项通常需要注意哪些流程？"),
    (re.compile(r"(图书馆|借阅|馆藏|数据库|座位|入馆)"), "清华图书馆相关服务有哪些要点？"),
    (re.compile(r"(奖学金|资助|助学|就业|招聘|实习)"), "关于学生发展或就业资助，有哪些公开信息？"),
]


def compact_answer(text: str, max_chars: int) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    cut = text.rfind("。", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[: cut + 1].strip()


def build_examples_for_page(page: dict, max_answer_chars: int) -> list[dict]:
    title = page.get("title") or "该页面"
    url = page.get("url", "")
    text = clean_text(page.get("text", ""))
    chunks = chunk_text(text)
    sentences = split_sentences(text)
    examples = []

    for idx, chunk in enumerate(chunks[:8]):
        template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        examples.append(
            {
                "instruction": template.format(title=title),
                "input": "",
                "output": compact_answer(chunk, max_answer_chars),
                "source_title": title,
                "source_url": url,
            }
        )

    keyword_answers = []
    for sentence in sentences:
        for pattern, question in KEYWORD_QUESTIONS:
            if pattern.search(sentence):
                keyword_answers.append((question.format(title=title), sentence))
                break
    keyword_answers = [(q, a) for q, a in keyword_answers if len(a) >= 18]
    seen_question_answer = set()
    for question, sentence in keyword_answers[:10]:
        key = stable_hash(question + sentence)
        if key in seen_question_answer:
            continue
        seen_question_answer.add(key)
        examples.append(
            {
                "instruction": question,
                "input": "",
                "output": compact_answer(sentence, max_answer_chars),
                "source_title": title,
                "source_url": url,
            }
        )

    return examples


def quality_filter(rows: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()
    for row in rows:
        instruction = re.sub(r"\s+", " ", row["instruction"]).strip()
        output = compact_answer(row["output"], 700)
        if len(instruction) < 6 or len(output) < 18:
            continue
        if "版权所有" in output and len(output) < 80:
            continue
        key = stable_hash(instruction + "\n" + output[:220])
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["instruction"] = instruction
        item["output"] = output
        cleaned.append(item)
    return cleaned


def add_ids(rows: list[dict]) -> list[dict]:
    result = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["id"] = f"thuqa-{idx:05d}"
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
                "source_url": row.get("source_url", ""),
                "source_title": row.get("source_title", ""),
            }
        )
    return converted


def build_dataset(args: argparse.Namespace) -> None:
    pages = list(read_jsonl(args.input))
    examples = []
    for page in tqdm(pages, desc="build qa"):
        examples.extend(build_examples_for_page(page, args.max_answer_chars))

    examples = quality_filter(examples)
    random.Random(args.seed).shuffle(examples)
    examples = add_ids(examples)

    if len(examples) < args.min_total:
        print(
            f"warning: only built {len(examples)} examples. "
            f"Add more seed URLs or increase --max-pages to reach {args.min_total}+."
        )

    eval_size = min(args.eval_size, max(1, len(examples) // 5))
    eval_rows = examples[:eval_size]
    train_rows = examples[eval_size:]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.all_output, examples)
    write_jsonl(args.sharegpt_output, convert_sharegpt(train_rows))

    print(f"pages: {len(pages)}")
    print(f"train: {len(train_rows)} -> {args.train_output}")
    print(f"eval:  {len(eval_rows)} -> {args.eval_output}")
    print(f"all:   {len(examples)} -> {args.all_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction QA data from crawled Tsinghua pages.")
    parser.add_argument("--input", default="data/raw/thu_pages.jsonl")
    parser.add_argument("--train-output", default="data/processed/train.jsonl")
    parser.add_argument("--eval-output", default="data/processed/eval.jsonl")
    parser.add_argument("--all-output", default="data/processed/qa_all.jsonl")
    parser.add_argument("--sharegpt-output", default="data/processed/train_sharegpt.jsonl")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--min-total", type=int, default=250)
    parser.add_argument("--max-answer-chars", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
