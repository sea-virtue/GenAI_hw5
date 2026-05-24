from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
import sys
from urllib.parse import urlparse

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.encoding_utils import looks_mojibake, repair_mojibake
from src.thu_qa.io_utils import read_jsonl, write_jsonl
from src.thu_qa.text_utils import chunk_text, clean_text, split_sentences, stable_hash


QUESTION_TEMPLATES = [
    "请概括介绍{title}。",
    "关于{title}，有哪些重要信息？",
    "如果同学想了解{title}，应该重点注意什么？",
]

PAGE_TYPE_QUESTIONS = {
    "calendar": [
        "清华大学校历页面提供了哪些学年或学期信息？",
        "同学在哪里可以查看清华大学校历相关安排？",
    ],
    "library": [
        "清华大学图书馆相关服务有哪些要点？",
        "同学使用清华大学图书馆服务时应注意哪些信息？",
    ],
    "traffic": [
        "清华大学校园交通或周边交通有哪些公开信息？",
        "到清华大学或在校内出行时，可以参考哪些交通信息？",
    ],
    "visit": [
        "公众参观清华大学校园需要注意哪些公开要求？",
        "清华大学校园参观如何预约或安排？",
    ],
    "contact": [
        "如何联系或咨询{title}相关事务？",
        "{title}公开了哪些联系方式或办公信息？",
    ],
    "admission_career": [
        "{title}提供了哪些招生、就业或培养相关信息？",
        "同学了解{title}时应重点关注哪些招生或发展信息？",
    ],
    "department_profile": [
        "请概括介绍{title}的基本情况。",
        "{title}有哪些历史、定位或发展特色？",
    ],
    "research_teaching": [
        "{title}在教学或科研方面有哪些公开信息？",
        "如果同学想了解{title}的教学科研情况，应该重点注意什么？",
    ],
    "table_or_list": [
        "根据公开信息，{title}列出了哪些主要内容？",
    ],
    "general": QUESTION_TEMPLATES,
}

KEYWORD_QUESTIONS = [
    (
        re.compile(r"(办公地点|通信地址|地址[:：]|位于|校区|校门|路线|交通|地铁|公交)"),
        "关于{title}的地点或交通信息，有哪些公开内容？",
        {"traffic", "visit", "contact", "general"},
    ),
    (
        re.compile(r"(开放时间|开馆时间|闭馆|服务时间|工作日|周一|周二|周三|周四|周五|上午|下午|校历|学期|假期|考试周|截止日期)"),
        "关于{title}的时间安排，有哪些公开信息？",
        {"calendar", "library", "traffic", "visit", "contact", "general"},
    ),
    (
        re.compile(r"(电话|邮箱|邮件|E-?MAIL|联系方式|联系电话|咨询电话|传真|邮政编码)"),
        "如何联系或咨询{title}相关事务？",
        {"contact", "library", "admission_career", "general"},
    ),
    (
        re.compile(r"(申请|办理|预约|注册|提交|流程|步骤|实名预约|入馆|借阅|座位预约)"),
        "办理或使用{title}相关服务时通常需要注意哪些流程？",
        {"library", "visit", "admission_career", "general"},
    ),
]

WEAK_TITLE_KEYWORDS = [
    "新闻",
    "动态",
    "通知",
    "公告",
    "学生风采",
    "教指委动态",
    "博士后招聘",
    "招聘",
    "讲座",
    "活动",
    "历任领导",
    "现任领导",
    "院士",
    "友情链接",
    "学生会",
    "团委",
    "委员会",
    "党政联席会",
    "学术组织",
    "历任",
    "名单",
    "学生组织",
]

PAGE_TYPE_HINTS = {
    "calendar": ["校历", "calendar"],
    "library": ["图书馆", "开馆", "借阅", "馆藏", "数据库", "座位", "lib.tsinghua"],
    "traffic": ["校园交通", "周边交通", "校车", "公交", "地铁", "到校路线"],
    "visit": ["校园参观", "参观清华", "参观预约"],
    "contact": ["联系方式", "联系电话", "邮箱", "办公地点", "地址", "联系我们"],
    "admission_career": ["招生", "就业", "培养方案", "职业发展", "申请", "专业介绍"],
    "research_teaching": ["科研", "教学", "课程", "研究方向", "培养方案", "专业介绍"],
    "department_profile": ["概况", "简介", "介绍", "历史沿革", "主任致辞", "学院介绍", "系所介绍"],
}


def compact_answer(text: str, max_chars: int) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    cut = text.rfind("。", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[: cut + 1].strip()


def page_blob(page: dict) -> str:
    return "\n".join([page.get("url", ""), page.get("title", ""), page.get("text", "")[:1000]])


def text_shape(text: str) -> dict[str, float]:
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    if not lines:
        return {"line_count": 0, "short_ratio": 1.0, "sentence_ratio": 0.0}
    sentence_lines = [line for line in lines if len(line) >= 20 and any(ch in line for ch in "。，；：？！")]
    short_lines = [line for line in lines if len(line) <= 12]
    return {
        "line_count": len(lines),
        "short_ratio": len(short_lines) / len(lines),
        "sentence_ratio": len(sentence_lines) / len(lines),
    }


def infer_page_kind(page: dict, title: str, text: str) -> str:
    url_title = (page.get("url", "") + "\n" + title).lower()
    blob = (url_title + "\n" + text[:800]).lower()
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["calendar"]):
        return "calendar"
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["traffic"]):
        return "traffic"
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["visit"]):
        return "visit"
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["contact"]):
        return "contact"
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["admission_career"]):
        return "admission_career"
    if any(hint.lower() in url_title for hint in PAGE_TYPE_HINTS["research_teaching"]):
        return "research_teaching"
    for kind, hints in PAGE_TYPE_HINTS.items():
        if kind in {"calendar", "traffic", "visit", "admission_career", "research_teaching"}:
            continue
        if any(hint.lower() in blob for hint in hints):
            return kind
    shape = text_shape(text)
    if shape["line_count"] >= 12 and shape["short_ratio"] > 0.65:
        return "table_or_list"
    host = urlparse(page.get("url", "")).netloc
    if host.endswith("tsinghua.edu.cn") and host not in {"www.tsinghua.edu.cn", "info.tsinghua.edu.cn"}:
        return "department_profile"
    return "general"


def weak_page(title: str, text: str, page_kind: str) -> bool:
    if page_kind in {"calendar", "library", "traffic", "visit", "contact"}:
        return False
    if any(keyword in title for keyword in WEAK_TITLE_KEYWORDS):
        return True
    shape = text_shape(text)
    if shape["line_count"] >= 20 and shape["short_ratio"] > 0.78 and shape["sentence_ratio"] < 0.12:
        return True
    first_lines = "\n".join(clean_text(text).splitlines()[:5])
    if re.search(r"(\d{4}[-./]\d{1,2}[-./]\d{1,2}|20\d{2}年\d{1,2}月|\d{1,2}-\d{1,2})", first_lines) and page_kind not in {
        "calendar",
        "library",
        "traffic",
    }:
        return True
    return False


def build_examples_for_page(page: dict, max_answer_chars: int) -> list[dict]:
    title = repair_mojibake(page.get("title") or "\u8be5\u9875\u9762")
    url = page.get("url", "")
    text = clean_text(repair_mojibake(page.get("text", "")))
    if looks_mojibake(title) or looks_mojibake(text):
        return []
    page_kind = infer_page_kind(page, title, text)
    if weak_page(title, text, page_kind):
        return []

    chunks = chunk_text(text)
    sentences = split_sentences(text)
    examples = []

    templates = PAGE_TYPE_QUESTIONS.get(page_kind, QUESTION_TEMPLATES)
    max_chunks = 2 if page_kind == "table_or_list" else 4
    for idx, chunk in enumerate(chunks[:max_chunks]):
        template = templates[idx % len(templates)]
        examples.append(
            {
                "instruction": template.format(title=title),
                "input": "",
                "output": compact_answer(chunk, max_answer_chars),
                "source_title": title,
                "source_url": url,
                "page_kind": page_kind,
            }
        )

    keyword_answers = []
    for sentence in sentences:
        for pattern, question, allowed_kinds in KEYWORD_QUESTIONS:
            if page_kind in allowed_kinds and pattern.search(sentence):
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
                "page_kind": page_kind,
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
        if any(keyword in instruction for keyword in ["学生风采", "博士后招聘", "教指委动态"]):
            continue
        if any(marker in output for marker in ["上页 下页", "/1页", "更多 >", "了解详细"]):
            continue
        if row.get("page_kind") == "table_or_list" and len(output) < 40:
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
                "page_kind": row.get("page_kind", ""),
            }
        )
    return converted


def split_train_eval(examples: list[dict], eval_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    preferred_eval_kinds = {
        "calendar",
        "library",
        "traffic",
        "visit",
        "contact",
        "admission_career",
        "department_profile",
        "research_teaching",
    }
    rng = random.Random(seed)
    preferred = [row for row in examples if row.get("page_kind") in preferred_eval_kinds]
    fallback = [row for row in examples if row.get("page_kind") not in preferred_eval_kinds]
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    selected_eval = preferred[:eval_size]
    if len(selected_eval) < eval_size:
        selected_eval.extend(fallback[: eval_size - len(selected_eval)])
    eval_keys = {row["id"] for row in selected_eval}
    train_rows = [row for row in examples if row["id"] not in eval_keys]
    return train_rows, selected_eval


def build_dataset(args: argparse.Namespace) -> None:
    pages = list(read_jsonl(args.input))
    examples = []
    used_pages = 0
    for page in tqdm(pages, desc="build qa"):
        if page.get("qa_candidate") is False and not args.include_low_quality:
            continue
        page_examples = build_examples_for_page(page, args.max_answer_chars)
        if page_examples:
            used_pages += 1
            examples.extend(page_examples)

    examples = quality_filter(examples)
    random.Random(args.seed).shuffle(examples)
    examples = add_ids(examples)

    if len(examples) < args.min_total:
        print(
            f"warning: only built {len(examples)} examples. "
            f"Add more seed URLs or increase --max-pages to reach {args.min_total}+."
        )

    eval_size = min(args.eval_size, max(1, len(examples) // 5))
    train_rows, eval_rows = split_train_eval(examples, eval_size, args.seed)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.all_output, examples)
    write_jsonl(args.sharegpt_output, convert_sharegpt(train_rows))

    print(f"pages: {len(pages)}")
    print(f"used_pages: {used_pages}")
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
    parser.add_argument("--min-total", type=int, default=600)
    parser.add_argument("--max-answer-chars", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-low-quality",
        action="store_true",
        help="Also build QA from raw pages marked qa_candidate=false. Not recommended for final training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
