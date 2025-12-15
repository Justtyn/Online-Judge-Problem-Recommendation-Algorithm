import argparse
import csv
import json
import os
from typing import Any


def read_allowed_tags(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "tag_name" not in reader.fieldnames:
            raise SystemExit(f"{path} missing column 'tag_name'")
        out: set[str] = set()
        for row in reader:
            v = (row.get("tag_name") or "").strip()
            if v:
                out.add(v)
    if not out:
        raise SystemExit(f"No tag_name loaded from {path}")
    return out


def parse_tags_cell(cell: str) -> list[str] | None:
    s = (cell or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            obj: Any = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
            return None
        return [x.strip() for x in obj if x.strip()]
    return [p.strip() for p in s.split(",") if p.strip()]


def unique_keep_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_int(s: str) -> int | None:
    try:
        return int(str(s).strip())
    except Exception:
        return None


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize CleanData/problems.csv tags into JSON arrays."
    )
    parser.add_argument("--problems", default="CleanData/problems.csv")
    parser.add_argument("--tags", default="CleanData/tags.csv", help="Tag whitelist CSV.")
    parser.add_argument("--output", default="CleanData/problems.json_tags.csv")
    parser.add_argument("--inplace", action="store_true", help="Overwrite --problems.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output.")
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    allowed = read_allowed_tags(args.tags)
    max_errors = max(1, int(args.max_errors))

    with open(args.problems, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{args.problems} has no header")
        fieldnames = list(reader.fieldnames)
        if "tags" not in fieldnames:
            raise SystemExit(f"{args.problems} missing column 'tags'")
        rows = list(reader)

    changed = 0
    errors: list[str] = []
    for idx, row in enumerate(rows, start=2):
        pid = (row.get("problem_id") or "").strip() or f"line {idx}"

        d = (row.get("difficulty") or "").strip()
        if d:
            v = parse_int(d)
            if v is None or not (1 <= v <= 10):
                errors.append(f"{args.problems}:{idx} problem_id={pid}: invalid difficulty {d!r}")
                if len(errors) >= max_errors:
                    break

        raw = (row.get("tags") or "").strip()
        if not raw:
            continue
        tags_list = parse_tags_cell(raw)
        if tags_list is None:
            errors.append(f"{args.problems}:{idx} problem_id={pid}: cannot parse tags {raw!r}")
            if len(errors) >= max_errors:
                break
            continue

        tags_list = unique_keep_order(tags_list)
        if not tags_list:
            errors.append(f"{args.problems}:{idx} problem_id={pid}: empty tags after parsing {raw!r}")
            if len(errors) >= max_errors:
                break
            continue

        bad = [t for t in tags_list if t not in allowed]
        if bad:
            errors.append(
                f"{args.problems}:{idx} problem_id={pid}: tags not in whitelist {bad!r} (raw={raw!r})"
            )
            if len(errors) >= max_errors:
                break
            continue

        normalized = json.dumps(tags_list, ensure_ascii=False, separators=(",", ":"))
        if normalized != raw:
            row["tags"] = normalized
            changed += 1

    if errors:
        for e in errors[:20]:
            print(e)
        print(f"FAILED: {len(errors)} errors (showing up to {max_errors}).")
        return 1

    out_path = args.problems if args.inplace else args.output
    if args.dry_run:
        print(f"OK (dry-run): would write {out_path}; changed={changed}; rows={len(rows)}")
        return 0

    write_csv(out_path, fieldnames, rows)
    print(f"OK: wrote {out_path}; changed={changed}; rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

