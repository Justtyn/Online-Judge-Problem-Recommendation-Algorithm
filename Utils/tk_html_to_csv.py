#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utils/tk_html_to_csv.py

用途
- 将题库网页（HTML）解析为结构化 CSV（题目列表/题面/样例/限制等），用于后续：
  - 生成标注请求（`Utils/csv_to_requests.py`）
  - 合并标注结果到 problems（`Utils/merge_labels_into_originaldata_problems.py`）

输入
- 题库 HTML 文件或目录（具体以 CLI 参数为准）

输出
- `tk_problems.csv`（默认）：包含题目基础信息与正文文本字段

说明
- 该脚本是“离线解析器”，不依赖网络；核心是把 HTML 转成干净的纯文本字段。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional


class _HTMLTextExtractor(HTMLParser):
    """将 HTML 片段抽取为较干净的纯文本（保留必要的换行/分隔符）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._chunks: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"br", "hr"}:
            self._chunks.append("\n")
        elif tag in {"p", "div", "tr", "li", "h1", "h2", "h3"}:
            self._chunks.append("\n")
        elif tag in {"td", "th"}:
            self._chunks.append("\t")
        elif tag == "pre":
            self._in_pre = True
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "pre":
            self._in_pre = False
            self._chunks.append("\n")
        elif tag in {"p", "div", "tr", "li", "h1", "h2", "h3"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._in_pre:
            self._chunks.append(data)
            return
        data = re.sub(r"\s+", " ", data)
        if data.strip():
            self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        self._chunks.append(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self._chunks.append(unescape(f"&#{name};"))

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = unescape(raw).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_fragment_to_text(fragment: str) -> str:
    """将 HTML 片段解析为纯文本（调用 _HTMLTextExtractor）。"""
    parser = _HTMLTextExtractor()
    parser.feed(fragment)
    return parser.text()


def _decode_html_bytes(data: bytes) -> str:
    """按常见编码（utf-8/utf-8-sig/gb18030）解码 HTML bytes。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_between(html: str, start_idx: int, end_idx: int) -> str:
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return ""
    return html[start_idx:end_idx]


def _extract_section_html(html: str, section_name: str) -> str:
    m = re.search(rf"<h2>\s*{re.escape(section_name)}\s*</h2>", html, flags=re.IGNORECASE)
    if not m:
        return ""
    after = html[m.end():]
    candidates: list[int] = []

    next_h2 = re.search(r"<h2>\s*[^<]+?\s*</h2>", after, flags=re.IGNORECASE)
    if next_h2:
        candidates.append(next_h2.start())

    end_mark = re.search(r"<!--\s*EndMarkForVirtualJudge\s*-->", after, flags=re.IGNORECASE)
    if end_mark:
        candidates.append(end_mark.start())

    if not candidates:
        return after
    return after[: min(candidates)]


def _extract_problem_title(html: str) -> str:
    m = re.search(r"<h2>\s*(\d+\s*:\s*[^<]+?)\s*</h2>", html)
    if not m:
        return ""
    return _html_fragment_to_text(m.group(1))


_TITLE_PREFIX_RE = re.compile(r"^\s*\d+\s*:\s*(.*?)\s*$", flags=re.DOTALL)


def _renumber_title(title: str, index: int) -> str:
    title = (title or "").strip()
    m = _TITLE_PREFIX_RE.match(title)
    rest = (m.group(1) if m else title).strip()
    if rest:
        return f"{index}: {rest}"
    return str(index)


def _extract_limits(html: str) -> tuple[str, str]:
    # Try extracting from the main <center> block which contains limits.
    center_start = html.find("<center>")
    mark = html.find("<!--StartMarkForVirtualJudge-->")
    header_block = ""
    if center_start != -1 and mark != -1 and mark > center_start:
        header_block = html[center_start:mark]
    header_text = _html_fragment_to_text(header_block) if header_block else _html_fragment_to_text(html[:3000])

    time_limit = ""
    memory_limit = ""

    tm = re.search(r"Time Limit:\s*([0-9.]+)\s*([A-Za-z]+)", header_text)
    if tm:
        time_limit = f"{tm.group(1)} {tm.group(2)}"

    mm = re.search(r"Memory Limit:\s*([0-9.]+)\s*([A-Za-z]+)", header_text)
    if mm:
        memory_limit = f"{mm.group(1)} {mm.group(2)}"

    return time_limit.strip(), memory_limit.strip()


@dataclass
class ProblemRow:
    title: str
    description: str
    sample_input: str
    sample_output: str
    hint: str
    source: str
    time_limit: str
    memory_limit: str
    difficulty: str = ""
    tags: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "sample_input": self.sample_input,
            "sample_output": self.sample_output,
            "hint": self.hint,
            "source": self.source,
            "time_limit": self.time_limit,
            "memory_limit": self.memory_limit,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }


def parse_problem_html(path: Path) -> ProblemRow:
    """解析单个题目 HTML 文件为结构化 ProblemRow。"""
    html = _decode_html_bytes(path.read_bytes())

    title = _extract_problem_title(html)
    time_limit, memory_limit = _extract_limits(html)

    description = _html_fragment_to_text(_extract_section_html(html, "Description"))
    sample_input = _html_fragment_to_text(_extract_section_html(html, "Sample Input"))
    sample_output = _html_fragment_to_text(_extract_section_html(html, "Sample Output"))
    hint = _html_fragment_to_text(_extract_section_html(html, "HINT"))
    source = _html_fragment_to_text(_extract_section_html(html, "Source"))

    return ProblemRow(
        title=title,
        description=description,
        sample_input=sample_input,
        sample_output=sample_output,
        hint=hint,
        source=source,
        time_limit=time_limit,
        memory_limit=memory_limit,
    )


def iter_html_files(input_dir: Path) -> Iterable[Path]:
    """遍历目录下的 .html 文件（按文件名排序，保证输出稳定）。"""

    def sort_key(path: Path):
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    for p in sorted(input_dir.glob("*.html"), key=sort_key):
        if p.is_file():
            yield p


def write_csv(rows: Iterable[ProblemRow], output_csv: Path) -> None:
    """写出题目列表 CSV（utf-8-sig，便于 Excel 直接打开）。"""
    fieldnames = [
        "title",
        "description",
        "sample_input",
        "sample_output",
        "hint",
        "source",
        "time_limit",
        "memory_limit",
        "difficulty",
        "tags",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def main() -> int:
    """CLI 入口：解析题库 HTML（文件或目录）并输出 tk_problems.csv。"""
    ap = argparse.ArgumentParser(description="Parse TK题库 HTML files into a CSV.")
    ap.add_argument("--input-dir", type=Path, default=Path("TK题库"), help="Directory containing .html files")
    ap.add_argument("--output", type=Path, default=Path("tk_problems.csv"), help="Output CSV path")
    ap.add_argument("--test-file", type=Path, default=None, help="Parse a single HTML and print JSON to stdout")
    ap.add_argument("--test-index", type=int, default=1, help="Index used when printing --test-file output")
    ap.add_argument("--limit", type=int, default=0, help="Only parse first N files (0 means all)")
    args = ap.parse_args()

    if args.test_file is not None:
        row = parse_problem_html(args.test_file)
        row.title = _renumber_title(row.title, args.test_index)
        print(json.dumps(row.as_dict(), ensure_ascii=False, indent=2))
        return 0

    rows: list[ProblemRow] = []
    for idx, path in enumerate(iter_html_files(args.input_dir), start=1):
        if args.limit and idx > args.limit:
            break
        row = parse_problem_html(path)
        row.title = _renumber_title(row.title, idx)
        rows.append(row)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
