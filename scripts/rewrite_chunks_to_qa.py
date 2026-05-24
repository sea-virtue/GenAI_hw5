from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thu_qa.encoding_utils import looks_mojibake, repair_mojibake
from src.thu_qa.text_utils import clean_text, split_sentences


SKIP_TITLE_KEYWORDS = {
    "新闻",
    "动态",
    "活动",
    "招聘",
    "讲座",
    "论坛",
    "团委",
    "党委",
    "工会",
    "党支部",
    "友情链接",
    "学生会",
    "校友访谈",
    "现任领导",
    "历任领导",
    "顾问委员会",
    "学术委员会",
    "科研项目",
    "成果展示",
    "学术活动",
    "科研成果",
    "全部",
}

USEFUL_TITLE_KEYWORDS = {
    "概况",
    "简介",
    "历史",
    "沿革",
    "教学",
    "科研",
    "研究",
    "招生",
    "培养",
    "专业",
    "学位",
    "博士",
    "硕士",
    "联系方式",
    "联系我们",
    "开馆时间",
    "规章制度",
    "校历",
    "校园参观",
    "校园交通",
    "周边交通",
    "统计资料",
    "院系设置",
    "组织机构",
    "服务信息",
    "实用信息",
    "图书馆",
}


def strip_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    for sep in ["-", "_", "－"]:
        if sep in title:
            head, tail = [part.strip() for part in title.split(sep, 1)]
            if head in {"联系我们", "联系方式"} and tail:
                return f"{tail}联系方式"
    for sep in ["-清华大学", "_清华大学", "－清华大学"]:
        if sep in title:
            title = title.split(sep, 1)[0].strip()
    return title or "该页面"


def compact(text: str, max_chars: int = 360) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = max(text.rfind("。", 0, max_chars), text.rfind("；", 0, max_chars))
    if cut < max_chars // 2:
        cut = max_chars
    return text[: cut + 1].strip()


def text_shape(text: str) -> tuple[int, float, float]:
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    if not lines:
        return 0, 1.0, 0.0
    short = sum(len(line) <= 12 for line in lines) / len(lines)
    sentence = sum(len(line) >= 20 and any(ch in line for ch in "。，；：？！") for line in lines) / len(lines)
    return len(lines), short, sentence


def should_skip(title: str, text: str, qa_candidate: bool) -> bool:
    if not title or not text or looks_mojibake(title) or looks_mojibake(text):
        return True
    if any(keyword in title for keyword in SKIP_TITLE_KEYWORDS):
        return True
    line_count, short_ratio, sentence_ratio = text_shape(text)
    if line_count >= 20 and short_ratio > 0.78 and sentence_ratio < 0.12:
        return True
    if qa_candidate:
        return False
    if any(keyword in title for keyword in USEFUL_TITLE_KEYWORDS):
        return False
    if re.search(r"\b\d{3,}|\w+@\w+", text):
        return False
    return True


def add(rows: list[dict], instruction: str, output: str) -> None:
    output = compact(output)
    if len(output) < 18:
        return
    rows.append({"instruction": instruction.strip(), "output": output})


def first_sentences(text: str, limit: int = 2) -> str:
    sentences = [s for s in split_sentences(text) if len(s) >= 18]
    return " ".join(sentences[:limit])


def contact_examples(title: str, text: str) -> list[dict]:
    rows: list[dict] = []
    if not re.search(r"(电话|邮箱|E-mail|联系方式|联系我们|@)", title + text, re.I):
        return rows

    if "图书馆" in title and "62784591" in text:
        add(rows, "清华大学图书馆总馆联系电话是多少？", "清华大学图书馆总馆联系电话为62784591/62782137。")
    if "图书馆" in title and "ref-desk@mail.tsinghua.edu.cn" in text:
        add(
            rows,
            "清华大学图书馆参考咨询的电话和邮箱是什么？",
            "清华大学图书馆参考咨询可联系总咨询台（在线），电话为62782137，邮箱为ref-desk@mail.tsinghua.edu.cn。",
        )
    if "图书馆" in title and "circdesk@tsinghua.edu.cn" in text:
        add(
            rows,
            "清华大学图书馆借还书等读者服务问题如何联系？",
            "借还书、研读间研讨间、座位管理系统等读者服务相关问题可联系北馆一层总服务台，电话为62788937，邮箱为circdesk@tsinghua.edu.cn。",
        )

    email_matches = sorted(set(re.findall(r"[\w.\-+]+@[\w.\-]+\.\w+", text)))
    phone_matches = sorted(set(re.findall(r"(?<!\d)(?:\(?010\)?[- ]?)?[268]\d{7}(?!\d)", text)))
    if not rows and (email_matches or phone_matches):
        parts = []
        if phone_matches:
            parts.append("电话：" + "、".join(phone_matches[:5]))
        if email_matches:
            parts.append("邮箱：" + "、".join(email_matches[:5]))
        add(rows, f"{title}公开了哪些联系方式？", "；".join(parts) + "。")
    return rows


def special_examples(title: str, text: str) -> list[dict]:
    rows: list[dict] = []
    if "院系设置" in title and "共设" in text:
        m = re.search(r"共设(\d+)个学院、(\d+)个系、(\d+)个书院", text)
        if m:
            add(
                rows,
                "清华大学目前设有多少个学院、系和书院？",
                f"清华大学目前共设{m.group(1)}个学院、{m.group(2)}个系、{m.group(3)}个书院，已成为一所设有理学、工学、文学、艺术学、历史学、哲学、经济学、管理学、法学、教育学、医学和交叉学科等12个学科门类的综合性、研究型、开放式大学。",
            )

    if "图书馆" in title and "8.44" in text and "625.07" in text:
        add(
            rows,
            "清华大学图书馆的建筑面积和馆藏量是多少？",
            "清华大学图书馆建筑面积为8.44万平方米，实体馆藏625.07万册（件），其中古籍线装书22.25万册。",
        )

    if "研究生招生" in title and all(keyword in text for keyword in ["医疗管理", "MHA", "临床医学"]):
        add(
            rows,
            "清华大学有哪些医疗管理相关的研究生项目？",
            "清华大学设有医疗管理硕士专业学位项目（MHA）、高级健康管理与转化医学（EMTM）硕士项目、临床医学专业学位研究生项目等医疗管理和医学培养相关项目。",
        )

    return rows


def generic_examples(title: str, text: str, qa_candidate: bool = True) -> list[dict]:
    rows: list[dict] = []
    sentences = [s for s in split_sentences(text) if len(s) >= 20]
    if not sentences:
        return rows
    first = sentences[0]
    if re.search(r"^(近日|日前|\d{4}年|\d{1,2}月|\w{3}\s+\d{1,2},\s+\d{4})", first):
        return rows
    if not qa_candidate:
        lead = text[:500]
        profileish = re.search(r"(成立于|创办|旨在|致力于|配套|采用|设有|下设|位于|建立起|发展成为)", lead)
        eventish = re.search(r"(预告|比赛时间|荣获|获批|新闻|讲座|论坛|研讨会|20\d{2}[./年]\d{1,2}|20\d{2}\.\d{1,2})", lead)
        if not profileish or (eventish and not re.search(r"(成立于|创办|旨在|致力于|发展成为)", lead)):
            return rows

    head = first_sentences(text, 2)
    if head:
        if re.search(r"(概况|简介|学院|书院|系|中心|研究院|实验室)$", title):
            add(rows, f"{title}的基本情况是什么？", head)
        else:
            add(rows, f"{title}主要介绍了什么？", head)

    patterns = [
        (r"(成立于|始建|创办|更名|历史|沿革|创建)", f"{title}有哪些历史沿革信息？"),
        (r"(设有|下设|包括|分为|平台|机构)", f"{title}包含哪些机构、平台或项目？"),
        (r"(招生|培养|学位|博士|硕士|课程|专业)", f"{title}有哪些人才培养或招生信息？"),
        (r"(科研|研究方向|成果|合作|项目)", f"{title}有哪些科研或学术信息？"),
        (r"(服务|办理|预约|开放|规则|制度)", f"{title}有哪些服务或规则信息？"),
    ]
    used = {row["output"] for row in rows}
    for pattern, question in patterns:
        matched = [s for s in sentences if re.search(pattern, s)]
        answer = " ".join(matched[:2])
        if answer and compact(answer) not in used:
            add(rows, question, answer)
            used.add(compact(answer))
        if len(rows) >= 3:
            break

    return rows


def page_to_qa(page: dict) -> list[dict]:
    title = strip_title(repair_mojibake(page.get("title", "")))
    text = clean_text(repair_mojibake(page.get("text", "")))
    qa_candidate = bool(page.get("qa_candidate"))
    if should_skip(title, text, qa_candidate):
        return []

    rows: list[dict] = []
    rows.extend(special_examples(title, text))
    rows.extend(contact_examples(title, text))
    if rows:
        return rows[:4]
    if not qa_candidate:
        return []
    rows.extend(generic_examples(title, text, qa_candidate=qa_candidate))

    seen = set()
    deduped = []
    for row in rows:
        key = (row["instruction"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:4]


def rewrite_file(path: Path, dry_run: bool = False) -> int:
    qa_rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        qa_rows.extend(page_to_qa(page))

    seen = set()
    deduped = []
    for row in qa_rows:
        key = row["instruction"] + "\n" + row["output"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if not dry_run:
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in deduped)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
    return len(deduped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite raw page chunk files into QA JSONL text files.")
    parser.add_argument("paths", nargs="+", help="Chunk txt files to rewrite in place.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    for item in args.paths:
        path = Path(item)
        count = rewrite_file(path, dry_run=args.dry_run)
        print(f"{path}: {count} QA pairs")


if __name__ == "__main__":
    main(parse_args())
