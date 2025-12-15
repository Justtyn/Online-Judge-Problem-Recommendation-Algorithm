import argparse
import csv
import json
import os
from collections.abc import Iterable
from typing import Any


def read_set(path: str, key: str) -> set[str]:
    out: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or key not in reader.fieldnames:
            raise SystemExit(f"{path} missing column {key!r}")
        for row in reader:
            v = (row.get(key) or "").strip()
            if v:
                out.add(v)
    return out


def iter_rows(path: str) -> Iterable[tuple[int, dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        for line_no, row in enumerate(reader, start=2):
            yield line_no, row


def parse_int(s: str) -> int | None:
    try:
        return int(str(s).strip())
    except Exception:
        return None


def validate_problems(
    *,
    problems_csv: str,
    allowed_tags: set[str],
    accept_csv_tags: bool,
    max_errors: int,
    errors: list[str],
) -> None:
    for line_no, row in iter_rows(problems_csv):
        pid = (row.get("problem_id") or "").strip() or f"line {line_no}"

        d = (row.get("difficulty") or "").strip()
        if d:
            v = parse_int(d)
            if v is None or not (1 <= v <= 10):
                errors.append(f"{problems_csv}:{line_no} problem_id={pid}: invalid difficulty {d!r}")
                if len(errors) >= max_errors:
                    return

        t = (row.get("tags") or "").strip()
        if not t:
            continue

        tags_list: list[str] | None = None
        if t.startswith("["):
            try:
                obj: Any = json.loads(t)
            except json.JSONDecodeError:
                errors.append(f"{problems_csv}:{line_no} problem_id={pid}: tags not valid JSON: {t!r}")
                if len(errors) >= max_errors:
                    return
                continue
            if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
                errors.append(
                    f"{problems_csv}:{line_no} problem_id={pid}: tags must be JSON array of strings: {t!r}"
                )
                if len(errors) >= max_errors:
                    return
                continue
            tags_list = [x.strip() for x in obj if x.strip()]
        elif accept_csv_tags:
            tags_list = [x.strip() for x in t.split(",") if x.strip()]
        else:
            errors.append(
                f"{problems_csv}:{line_no} problem_id={pid}: tags must be JSON array (got non-JSON): {t!r}"
            )
            if len(errors) >= max_errors:
                return
            continue

        if not tags_list:
            errors.append(f"{problems_csv}:{line_no} problem_id={pid}: empty tags")
            if len(errors) >= max_errors:
                return
            continue

        bad = [x for x in tags_list if x not in allowed_tags]
        if bad:
            errors.append(
                f"{problems_csv}:{line_no} problem_id={pid}: tags not in whitelist: {bad!r} (raw={t!r})"
            )
            if len(errors) >= max_errors:
                return


def validate_submissions(
    *,
    submissions_csv: str,
    students_user_ids: set[str],
    problems_ids: set[str],
    language_names: set[str],
    verdict_names: set[str],
    max_errors: int,
    errors: list[str],
) -> None:
    last_attempt: dict[tuple[str, str], int] = {}
    for line_no, row in iter_rows(submissions_csv):
        sid = (row.get("submission_id") or "").strip() or f"line {line_no}"

        user_id = (row.get("user_id") or "").strip()
        if not user_id or user_id not in students_user_ids:
            errors.append(
                f"{submissions_csv}:{line_no} submission_id={sid}: user_id {user_id!r} not in students.user_id"
            )
            if len(errors) >= max_errors:
                return

        problem_id = (row.get("problem_id") or "").strip()
        if not problem_id or problem_id not in problems_ids:
            errors.append(
                f"{submissions_csv}:{line_no} submission_id={sid}: problem_id {problem_id!r} not in problems.problem_id"
            )
            if len(errors) >= max_errors:
                return

        language = (row.get("language") or "").strip()
        if not language or language not in language_names:
            errors.append(
                f"{submissions_csv}:{line_no} submission_id={sid}: language {language!r} not in languages.name"
            )
            if len(errors) >= max_errors:
                return

        verdict = (row.get("verdict") or "").strip()
        if not verdict or verdict not in verdict_names:
            errors.append(
                f"{submissions_csv}:{line_no} submission_id={sid}: verdict {verdict!r} not in verdicts.name"
            )
            if len(errors) >= max_errors:
                return

        ac_s = (row.get("ac") or "").strip()
        ac = parse_int(ac_s)
        if ac not in (0, 1):
            errors.append(f"{submissions_csv}:{line_no} submission_id={sid}: invalid ac {ac_s!r}")
            if len(errors) >= max_errors:
                return
        else:
            if (ac == 1) != (verdict == "AC"):
                errors.append(
                    f"{submissions_csv}:{line_no} submission_id={sid}: ac=={ac} but verdict=={verdict!r} (must be equivalent)"
                )
                if len(errors) >= max_errors:
                    return

        attempt_no_s = (row.get("attempt_no") or "").strip()
        attempt_no = parse_int(attempt_no_s)
        if attempt_no is None or attempt_no <= 0:
            errors.append(
                f"{submissions_csv}:{line_no} submission_id={sid}: invalid attempt_no {attempt_no_s!r}"
            )
            if len(errors) >= max_errors:
                return
        else:
            if user_id and problem_id:
                key = (user_id, problem_id)
                prev = last_attempt.get(key)
                if prev is not None and attempt_no <= prev:
                    errors.append(
                        f"{submissions_csv}:{line_no} submission_id={sid}: attempt_no out of order for (user_id,problem_id)={key}: {attempt_no} after {prev}"
                    )
                    if len(errors) >= max_errors:
                        return
                last_attempt[key] = attempt_no


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CleanData CSV integrity.")
    parser.add_argument("--students", default="CleanData/students.csv")
    parser.add_argument("--problems", default="CleanData/problems.csv")
    parser.add_argument("--submissions", default="CleanData/submissions.csv")
    parser.add_argument("--languages", default="CleanData/languages.csv")
    parser.add_argument("--verdicts", default="CleanData/verdicts.csv")
    parser.add_argument("--tags", default="CleanData/tags.csv")
    parser.add_argument(
        "--accept-csv-tags",
        action="store_true",
        help="Accept comma-separated tags in problems.tags (otherwise require JSON array).",
    )
    parser.add_argument("--max-errors", type=int, default=200, help="Stop after N errors.")
    parser.add_argument("--report", default="", help="Write errors to this file.")
    args = parser.parse_args()

    max_errors = max(1, int(args.max_errors))
    errors: list[str] = []

    students_user_ids = read_set(args.students, "user_id")
    problems_ids = read_set(args.problems, "problem_id")
    language_names = read_set(args.languages, "name")
    verdict_names = read_set(args.verdicts, "name")
    allowed_tags = read_set(args.tags, "tag_name")

    validate_submissions(
        submissions_csv=args.submissions,
        students_user_ids=students_user_ids,
        problems_ids=problems_ids,
        language_names=language_names,
        verdict_names=verdict_names,
        max_errors=max_errors,
        errors=errors,
    )
    if len(errors) < max_errors:
        validate_problems(
            problems_csv=args.problems,
            allowed_tags=allowed_tags,
            accept_csv_tags=bool(args.accept_csv_tags),
            max_errors=max_errors,
            errors=errors,
        )

    if args.report:
        report_path = args.report
        if not os.path.isabs(report_path):
            report_path = os.path.abspath(report_path)
        with open(report_path, "w", encoding="utf-8", newline="\n") as f:
            for e in errors:
                f.write(e)
                f.write("\n")
        print(f"Wrote report: {report_path}")

    if errors:
        print(f"FAILED: {len(errors)} issues found (showing up to {max_errors}).")
        for e in errors[: min(len(errors), 20)]:
            print(e)
        return 1

    print("OK: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

