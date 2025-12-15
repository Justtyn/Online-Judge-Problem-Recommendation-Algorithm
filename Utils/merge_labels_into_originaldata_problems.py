import argparse
import csv
import json
import os
import random
import re
from typing import Any


TITLE_ID_RE = re.compile(r"^\s*(\d+)\s*[:：]")


def _read_csv_dicts(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_csv_dicts(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _parse_problem_id(labeled_row: dict[str, str]) -> str | None:
    for key in ("problem_id", "id", "pid"):
        v = (labeled_row.get(key) or "").strip()
        if v:
            return v

    title = (labeled_row.get("title") or "").strip()
    if not title:
        return None
    m = TITLE_ID_RE.match(title)
    if not m:
        return None
    return m.group(1)


def _format_tags(tags_cell: str) -> str:
    s = (tags_cell or "").strip()
    if not s:
        return ""

    if s.startswith("["):
        try:
            v: Any = json.loads(s)
        except json.JSONDecodeError:
            return s
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return ",".join(v)
        return s

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ",".join(parts)


def _load_allowed_tags(path: str, column: str | None) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"Allowed-tags CSV has no header: {path}")

        chosen = (column or "").strip()
        if not chosen:
            for cand in ("tag_name", "tag", "name"):
                if cand in fieldnames:
                    chosen = cand
                    break
        if not chosen or chosen not in fieldnames:
            raise SystemExit(
                f"Cannot find tag column in {path}; pass --allowed-tags-column. "
                f"Available columns: {fieldnames}"
            )

        tags: list[str] = []
        for row in reader:
            v = (row.get(chosen) or "").strip()
            if v:
                tags.append(v)
    if not tags:
        raise SystemExit(f"No tags loaded from {path}")
    return tags


def _parse_tags_to_list(tags_cell: str) -> list[str]:
    s = (tags_cell or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            v: Any = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return [x.strip() for x in v if x.strip()]
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _difficulty_ok(s: str) -> bool:
    try:
        v = int(str(s).strip())
    except Exception:
        return False
    return 1 <= v <= 10


def _fill_missing_labels_in_row(
    row: dict[str, str],
    *,
    rng: random.Random,
    allowed_tags: list[str],
    min_tags: int,
    max_tags: int,
) -> bool:
    changed = False

    d = (row.get("difficulty") or "").strip()
    if not d:
        row["difficulty"] = str(rng.randint(1, 10))
        changed = True

    t = (row.get("tags") or "").strip()
    if not t:
        k = rng.randint(max(1, int(min_tags)), max(1, int(max_tags)))
        k = max(1, min(k, len(allowed_tags)))
        picks = rng.sample(allowed_tags, k=k)
        row["tags"] = json.dumps(picks, ensure_ascii=False, separators=(",", ":"))
        changed = True

    return changed


def _validate_rows(
    rows: list[dict[str, str]],
    *,
    allowed_tag_set: set[str],
    max_tags: int,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    for idx, r in enumerate(rows, start=1):
        pid = (r.get("problem_id") or "").strip() or f"row#{idx}"

        d = (r.get("difficulty") or "").strip()
        if not _difficulty_ok(d):
            errors.append(f"{pid}: invalid difficulty {d!r}")

        tags = (r.get("tags") or "").strip()
        parts = [p.strip() for p in tags.split(",") if p.strip()]
        if not parts:
            errors.append(f"{pid}: empty tags")
            continue
        if len(parts) > max_tags:
            errors.append(f"{pid}: too many tags ({len(parts)}): {tags!r}")
        if len(set(parts)) != len(parts):
            errors.append(f"{pid}: duplicate tags: {tags!r}")
        bad = [p for p in parts if p not in allowed_tag_set]
        if bad:
            errors.append(f"{pid}: invalid tags {bad!r} (raw={tags!r})")

    return len(errors), errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge labeled difficulty/tags into OriginalData/problems.csv"
    )
    parser.add_argument(
        "--labeled",
        default="tk_problems_labeled.csv",
        help="Input labeled CSV (difficulty,tags).",
    )
    parser.add_argument(
        "--problems",
        default="OriginalData/problems.csv",
        help="Original problems CSV to update.",
    )
    parser.add_argument(
        "--match",
        choices=("order", "problem_id"),
        default="order",
        help="How to match rows: by order (default) or by problem_id.",
    )
    parser.add_argument(
        "--allowed-tags-csv",
        default="OriginalData/tags.csv",
        help="CSV providing the allowed tag vocabulary (default: OriginalData/tags.csv).",
    )
    parser.add_argument(
        "--allowed-tags-column",
        default="",
        help="Column name in --allowed-tags-csv (auto: tag_name/tag/name).",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="If a labeled row has empty difficulty/tags, fill it randomly before merging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --fill-missing.",
    )
    parser.add_argument(
        "--min-tags",
        type=int,
        default=1,
        help="Min number of tags when filling missing tags.",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=2,
        help="Max number of tags when filling missing tags, and for validation.",
    )
    parser.add_argument(
        "--write-fixed-labeled",
        default="",
        help="If set, write a fixed labeled CSV (after --fill-missing) to this path.",
    )
    parser.add_argument(
        "--inplace-labeled",
        action="store_true",
        help="Overwrite --labeled with the fixed labeled CSV (implies --fill-missing).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip final full-file validation of difficulty/tags.",
    )
    parser.add_argument(
        "--validate-report",
        default="",
        help="If set, write validation errors to this text file.",
    )
    parser.add_argument(
        "--output",
        default="OriginalData/problems_labeled.csv",
        help="Output CSV path (default writes a new file).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite --problems (atomic replace).",
    )
    parser.add_argument(
        "--only-empty",
        action="store_true",
        help="Only fill rows where problems.csv difficulty/tags are empty.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any problem_id in problems.csv is missing in labeled CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output; just report match stats.",
    )
    args = parser.parse_args()

    labeled_fields, labeled_rows = _read_csv_dicts(args.labeled)
    if "difficulty" not in labeled_fields or "tags" not in labeled_fields:
        raise SystemExit(f"{args.labeled} must contain difficulty and tags columns.")

    problems_fields, problems_rows = _read_csv_dicts(args.problems)
    if "problem_id" not in problems_fields:
        raise SystemExit(f"{args.problems} must contain problem_id column.")
    if "difficulty" not in problems_fields:
        problems_fields.append("difficulty")
    if "tags" not in problems_fields:
        problems_fields.append("tags")

    allowed_tags = _load_allowed_tags(args.allowed_tags_csv, args.allowed_tags_column or None)
    allowed_tag_set = set(allowed_tags)

    matched = 0
    missing = 0
    skipped_only_empty = 0

    bad_labeled = 0
    labeled_by_id: dict[str, dict[str, str]] = {}
    fixed_labeled = 0

    if args.inplace_labeled:
        args.fill_missing = True

    if args.fill_missing:
        rng = random.Random(int(args.seed))
        for r in labeled_rows:
            changed = _fill_missing_labels_in_row(
                r,
                rng=rng,
                allowed_tags=allowed_tags,
                min_tags=args.min_tags,
                max_tags=args.max_tags,
            )
            if changed:
                fixed_labeled += 1

    if args.inplace_labeled or args.write_fixed_labeled:
        out_labeled = args.labeled if args.inplace_labeled else args.write_fixed_labeled
        if not out_labeled:
            raise SystemExit("Internal error: missing labeled output path.")
        if not args.dry_run:
            _write_csv_dicts(out_labeled, labeled_fields, labeled_rows)

    if args.match == "problem_id":
        for src in labeled_rows:
            pid = _parse_problem_id(src)
            if not pid:
                bad_labeled += 1
                continue
            labeled_by_id[pid] = src

        for r in problems_rows:
            pid = (r.get("problem_id") or "").strip()
            if not pid:
                continue

            src = labeled_by_id.get(pid)
            if not src:
                missing += 1
                continue

            if args.only_empty:
                if (r.get("difficulty") or "").strip() and (r.get("tags") or "").strip():
                    skipped_only_empty += 1
                    continue

            difficulty = (src.get("difficulty") or "").strip()
            tags = _format_tags(src.get("tags") or "")

            if difficulty:
                r["difficulty"] = difficulty
            if tags or not args.only_empty:
                r["tags"] = tags
            matched += 1
    else:
        if len(problems_rows) != len(labeled_rows):
            raise SystemExit(
                f"Order match requires same row count: problems={len(problems_rows)} labeled={len(labeled_rows)}"
            )
        for i, r in enumerate(problems_rows):
            src = labeled_rows[i]
            if args.only_empty:
                if (r.get("difficulty") or "").strip() and (r.get("tags") or "").strip():
                    skipped_only_empty += 1
                    continue

            difficulty = (src.get("difficulty") or "").strip()
            tags = _format_tags(src.get("tags") or "")

            if difficulty:
                r["difficulty"] = difficulty
            if tags or not args.only_empty:
                r["tags"] = tags
            matched += 1

    if args.strict and missing:
        raise SystemExit(
            f"Strict mode: {missing} problems missing labels (labeled loaded={len(labeled_by_id)})."
        )

    validate_errors = 0
    if not args.no_validate:
        validate_errors, error_lines = _validate_rows(
            problems_rows, allowed_tag_set=allowed_tag_set, max_tags=int(args.max_tags)
        )
        if args.validate_report:
            if not args.dry_run:
                with open(args.validate_report, "w", encoding="utf-8", newline="\n") as f:
                    for line in error_lines:
                        f.write(line)
                        f.write("\n")
        if validate_errors:
            raise SystemExit(
                f"Validation failed: {validate_errors} issues found. "
                f"{'See ' + args.validate_report if args.validate_report else 'Use --validate-report to save details.'}"
            )

    out_path = args.problems if args.inplace else args.output
    if not args.dry_run:
        _write_csv_dicts(out_path, problems_fields, problems_rows)

    print(
        f"Match={args.match}; labeled_rows={len(labeled_rows)} (bad_labeled={bad_labeled}); "
        f"fixed_labeled={fixed_labeled}; matched={matched}; missing={missing}; skipped_only_empty={skipped_only_empty}; "
        f"wrote={'(dry-run)' if args.dry_run else out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
