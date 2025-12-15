import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request


ALLOWED_TAGS = {
    "array_string",
    "hash_map",
    "two_pointers_sliding",
    "stack_queue",
    "linked_list",
    "tree",
    "graph",
    "dfs_bfs",
    "dp",
    "greedy_sorting",
    "binary_search",
    "math_bit",
}

BEGIN_RE = re.compile(r"^<<<BEGIN:(request-(\d+))>>>$")
END_RE = re.compile(r"^<<<END:(request-(\d+))>>>$")


class LabelParseError(ValueError):
    pass


def iter_prompt_blocks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        begin = BEGIN_RE.match(line)
        if not begin:
            raise ValueError(
                f"Invalid prompt file format at line {i+1}: expected BEGIN marker, got: {lines[i]!r}"
            )

        request_id = begin.group(1)
        request_num = int(begin.group(2))

        content_lines: list[str] = []
        i += 1
        while i < len(lines):
            maybe_end = END_RE.match(lines[i].strip())
            if maybe_end:
                end_id = maybe_end.group(1)
                end_num = int(maybe_end.group(2))
                if end_id != request_id or end_num != request_num:
                    raise ValueError(
                        f"Mismatched END marker at line {i+1}: {lines[i]!r} (expected <<<END:{request_id}>>>)"
                    )
                break
            content_lines.append(lines[i])
            i += 1

        if i >= len(lines):
            raise ValueError(f"Missing END marker for {request_id}")

        prompt = "\n".join(content_lines).strip("\n")
        yield request_num, request_id, prompt
        i += 1


def load_results(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    results: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in results file at line {line_no}: {e}") from e
            i = obj.get("i")
            if not isinstance(i, int) or i <= 0:
                raise ValueError(f"Invalid 'i' in results file at line {line_no}")
            results[i] = obj
    return results


def append_result(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return s


def parse_label(text: str) -> tuple[int, list[str]]:
    s = strip_code_fences(text)
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LabelParseError(f"Model output does not contain a JSON object: {text!r}")
    candidate = s[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LabelParseError(f"Model output JSON decode error: {e}") from e

    if set(obj.keys()) != {"difficulty", "tags"}:
        raise LabelParseError(
            f"Model output JSON must contain only difficulty and tags: {obj!r}"
        )

    difficulty = obj["difficulty"]
    tags = obj["tags"]
    if not isinstance(difficulty, int) or not (1 <= difficulty <= 10):
        raise LabelParseError(f"Invalid difficulty: {difficulty!r}")
    if not isinstance(tags, list) or not (1 <= len(tags) <= 2):
        raise LabelParseError(f"Invalid tags list: {tags!r}")
    for tag in tags:
        if not isinstance(tag, str) or tag not in ALLOWED_TAGS:
            raise LabelParseError(f"Invalid tag: {tag!r}")

    return difficulty, tags


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    system: str,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    data = json.loads(raw)
    return data["choices"][0]["message"]["content"]


def call_with_retries(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    system: str,
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                system=system,
                timeout_s=timeout_s,
            )
        except urllib.error.HTTPError as e:
            last_err = e
            status = getattr(e, "code", None)
            retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = "<no-body>"
                raise RuntimeError(f"HTTP {status}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt >= max_retries:
                raise RuntimeError(f"Network error: {e}") from e

        time.sleep(backoff_s * (2**attempt))

    raise RuntimeError(f"Failed after retries: {last_err}")


def write_csv(rows: list[dict], fieldnames: list[str], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def build_repair_prompt(original_prompt: str, bad_output: str) -> str:
    allowed = ", ".join(sorted(ALLOWED_TAGS))
    return (
        original_prompt
        + "\n\n【系统纠错】你刚才的输出不符合要求（例如包含不在可选列表内的标签、或不是一行 JSON）。\n"
        + "【可选标签列表】（只能从中选择，不可自造，不可变形）：\n"
        + allowed
        + "\n\n【要求重申】只允许输出一行 JSON，只能包含 difficulty,tags 两个键；difficulty 为 1-10 整数；tags 为 1-2 个且必须来自可选列表；不得输出任何其它字符。\n"
        + "你刚才的输出如下（仅供纠错参考）：\n"
        + bad_output.strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch label OJ problems with Qwen.")
    parser.add_argument("--prompts", default="请求.txt", help="Prompt file with BEGIN/END markers")
    parser.add_argument("--csv", default="tk_problems.csv", help="Input CSV path")
    parser.add_argument(
        "--output-csv", default="tk_problems_labeled.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write results back to --csv (overwrites it via atomic replace)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="DashScope API key (or set env DASHSCOPE_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Beijing region base_url",
    )
    parser.add_argument("--model", default="qwen3-max", help="Model name")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    parser.add_argument("--backoff", type=float, default=1.0, help="Retry backoff base seconds")
    parser.add_argument(
        "--repair-retries",
        type=int,
        default=2,
        help="If model output is invalid, retry with an auto-repair prompt N times",
    )
    parser.add_argument(
        "--results",
        default="labels_results.jsonl",
        help="Checkpoint results JSONL (append-only)",
    )
    parser.add_argument(
        "--errors",
        default="labels_errors.jsonl",
        help="Skipped/failed items JSONL (append-only)",
    )
    parser.add_argument("--start", type=int, default=1, help="Start index (1-based)")
    parser.add_argument("--end", type=int, default=0, help="End index (0=auto)")
    parser.add_argument("--index", type=int, default=0, help="Only run a single index (1-based)")
    parser.add_argument("--save-every", type=int, default=200, help="Write CSV every N updates")
    parser.add_argument(
        "--skip-filled",
        action="store_true",
        help="Skip rows that already have difficulty and tags in CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prompt/CSV alignment without calling the API",
    )
    args = parser.parse_args()

    prompts = list(iter_prompt_blocks(args.prompts))
    if not prompts:
        raise SystemExit("No prompts found in prompt file.")

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) != len(prompts):
        raise SystemExit(
            f"CSV rows ({len(rows)}) != prompt blocks ({len(prompts)}); cannot align."
        )

    if "difficulty" not in fieldnames:
        fieldnames.append("difficulty")
    if "tags" not in fieldnames:
        fieldnames.append("tags")

    if args.index:
        start = end = args.index
    else:
        start = max(1, args.start)
        end = args.end if args.end else len(rows)
        end = min(end, len(rows))

    if start > end:
        raise SystemExit(f"Invalid range: start={start} > end={end}")

    if args.inplace:
        out_csv = args.csv
    else:
        out_csv = args.output_csv
        if not os.path.isabs(out_csv):
            out_csv = os.path.join(os.path.dirname(os.path.abspath(args.csv)), out_csv)

    results = load_results(args.results)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    indices = list(range(start, end + 1))
    total = len(indices)
    if tqdm:
        it = tqdm(indices, total=total)
    else:
        it = indices

    if args.dry_run:
        for i in it:
            _num, request_id, prompt = prompts[i - 1]
            if request_id != f"request-{i}":
                raise SystemExit(f"Prompt id mismatch at {i}: got {request_id}")
            if not prompt.strip():
                raise SystemExit(f"Empty prompt at {i}")
        print(f"OK: validated {len(indices)} prompts against CSV rows.")
        return 0

    if not args.api_key:
        raise SystemExit("Missing API key: pass --api-key or set env DASHSCOPE_API_KEY.")

    system_prompt = "你是一个严格的输出器，只能输出用户要求的那一行 JSON，不能输出任何其它字符。"

    updated = 0
    for pos, i in enumerate(it, start=1):
        if not tqdm:
            width = 30
            filled = int(width * pos / total) if total else width
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(f"\r[{bar}] {pos}/{total}")
            sys.stderr.flush()

        row = rows[i - 1]
        _num, request_id, prompt = prompts[i - 1]

        if request_id != f"request-{i}":
            raise SystemExit(f"Prompt id mismatch at {i}: got {request_id}")

        if args.skip_filled and (row.get("difficulty") or "").strip() and (row.get("tags") or "").strip():
            continue

        if i in results:
            difficulty = results[i]["difficulty"]
            tags = results[i]["tags"]
        else:
            content = ""
            try:
                content = call_with_retries(
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    system=system_prompt,
                    timeout_s=args.timeout,
                    max_retries=args.max_retries,
                    backoff_s=args.backoff,
                )
                try:
                    difficulty, tags = parse_label(content)
                except LabelParseError:
                    repaired_output = content
                    for _ in range(max(0, args.repair_retries)):
                        repaired_prompt = build_repair_prompt(prompt, repaired_output)
                        repaired_output = call_with_retries(
                            api_key=args.api_key,
                            base_url=args.base_url,
                            model=args.model,
                            prompt=repaired_prompt,
                            system=system_prompt,
                            timeout_s=args.timeout,
                            max_retries=args.max_retries,
                            backoff_s=args.backoff,
                        )
                        try:
                            difficulty, tags = parse_label(repaired_output)
                            content = repaired_output
                            break
                        except LabelParseError:
                            continue
                    else:
                        raise

                append_result(
                    args.results,
                    {
                        "i": i,
                        "id": request_id,
                        "difficulty": difficulty,
                        "tags": tags,
                        "raw": content,
                    },
                )
            except Exception as e:
                append_result(
                    args.errors,
                    {
                        "i": i,
                        "id": request_id,
                        "error": str(e),
                        "raw": content,
                    },
                )
                continue

        row["difficulty"] = str(int(difficulty))
        row["tags"] = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
        updated += 1

        if args.index:
            print(json.dumps({"difficulty": int(difficulty), "tags": tags}, ensure_ascii=False))
            break

        if args.save_every > 0 and updated % args.save_every == 0:
            write_csv(rows, fieldnames, out_csv)

    if not tqdm:
        sys.stderr.write("\n")
    write_csv(rows, fieldnames, out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
