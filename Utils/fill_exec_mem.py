import argparse
import csv
import os
import random
import re
import sys
import time


TIME_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?")
MEM_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?")


def parse_time_limit_ms(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 1000
    m = TIME_RE.search(s)
    if not m:
        return 1000
    value = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit in {"ms", "msec", "msecs", "millisecond", "milliseconds"}:
        ms = value
    else:
        ms = value * 1000.0
    ms_int = int(round(ms))
    return max(1, ms_int)


def parse_memory_limit_kb(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 128 * 1024
    m = MEM_RE.search(s)
    if not m:
        return 128 * 1024
    value = float(m.group(1))
    unit = (m.group(2) or "mb").lower()
    if unit in {"kb", "k", "kib"}:
        kb = value
    elif unit in {"gb", "g", "gib"}:
        kb = value * 1024.0 * 1024.0
    else:
        kb = value * 1024.0
    kb_int = int(round(kb))
    return max(1, kb_int)


def clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def pick_within_limit(rng: random.Random, limit: int, *, lo_ratio: float, hi_ratio: float) -> int:
    if limit <= 1:
        return 1
    lo = max(1, int(limit * lo_ratio))
    hi = max(lo, int(limit * hi_ratio))
    if hi >= limit:
        hi = limit - 1
    return rng.randint(lo, max(lo, hi))


def pick_over_limit(rng: random.Random, limit: int, *, lo_ratio: float, hi_ratio: float) -> int:
    lo = max(limit + 1, int(limit * lo_ratio))
    hi = max(lo, int(limit * hi_ratio))
    return rng.randint(lo, hi)


def load_problem_limits(problems_csv: str) -> dict[int, tuple[int, int]]:
    limits: dict[int, tuple[int, int]] = {}
    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for idx, row in enumerate(r, start=1):
            pid_raw = (row.get("problem_id") or "").strip()
            if pid_raw.isdigit():
                pid = int(pid_raw)
            else:
                pid = idx
            tl_ms = parse_time_limit_ms(row.get("time_limit") or "")
            ml_kb = parse_memory_limit_kb(row.get("memory_limit") or "")
            limits[pid] = (tl_ms, ml_kb)
    return limits


def generate_exec_mem(
    *,
    rng: random.Random,
    verdict: str,
    time_limit_ms: int,
    memory_limit_kb: int,
) -> tuple[int, int]:
    verdict = (verdict or "").strip().upper()
    tl = max(1, time_limit_ms)
    ml = max(1, memory_limit_kb)

    if verdict == "CE":
        return 0, 0

    if verdict == "TLE":
        exec_time = pick_over_limit(rng, tl, lo_ratio=1.05, hi_ratio=3.00)
        mem_kb = pick_within_limit(rng, ml, lo_ratio=0.05, hi_ratio=0.90)
        return exec_time, mem_kb

    if verdict == "RE":
        exec_time = pick_within_limit(rng, tl, lo_ratio=0.01, hi_ratio=0.60)
        mem_kb = pick_within_limit(rng, ml, lo_ratio=0.05, hi_ratio=0.95)
        return exec_time, mem_kb

    # AC / WA and any other non-TLE, non-CE verdicts: within limits.
    exec_time = pick_within_limit(rng, tl, lo_ratio=0.05, hi_ratio=0.95)
    mem_kb = pick_within_limit(rng, ml, lo_ratio=0.05, hi_ratio=0.95)
    return exec_time, mem_kb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill exec_time_ms and mem_kb in submissions.csv using each problem's time/memory limits."
    )
    parser.add_argument(
        "--problems",
        default="OriginalData/problems.csv",
        help="Problems CSV containing problem_id,time_limit,memory_limit",
    )
    parser.add_argument(
        "--submissions",
        default="OriginalData/submissions.csv",
        help="Submissions CSV containing problem_id,verdict,exec_time_ms,mem_kb",
    )
    parser.add_argument(
        "--output",
        default="OriginalData/submissions_filled.csv",
        help="Output CSV path (ignored if --inplace)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite submissions CSV (atomic replace)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only validate inputs and show a small preview plan",
    )
    parser.add_argument("--progress-every", type=int, default=5000, help="Progress log interval")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    limits = load_problem_limits(args.problems)

    out_path = args.submissions if args.inplace else args.output
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)

    with open(args.submissions, "r", encoding="utf-8-sig", newline="") as f_in:
        r = csv.DictReader(f_in)
        if not r.fieldnames:
            raise SystemExit("submissions.csv missing header")
        fieldnames = list(r.fieldnames)
        if "exec_time_ms" not in fieldnames or "mem_kb" not in fieldnames:
            raise SystemExit("submissions.csv must contain exec_time_ms and mem_kb columns")
        if "problem_id" not in fieldnames or "verdict" not in fieldnames:
            raise SystemExit("submissions.csv must contain problem_id and verdict columns")

        if args.dry_run:
            # Read a few rows to validate mapping.
            preview = []
            for _ in range(5):
                try:
                    preview.append(next(r))
                except StopIteration:
                    break
            for row in preview:
                pid = int((row.get("problem_id") or "0").strip() or "0")
                if pid not in limits:
                    raise SystemExit(f"problem_id {pid} not found in problems.csv")
            print(
                f"OK dry-run: problems={len(limits)} submissions_preview={len(preview)} output={out_path}"
            )
            return 0

        tmp = out_path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()

            start = time.time()
            count = 0
            for row in r:
                count += 1
                pid = int((row.get("problem_id") or "0").strip() or "0")
                tl_ms, ml_kb = limits.get(pid, (1000, 128 * 1024))
                exec_ms, mem_kb = generate_exec_mem(
                    rng=rng,
                    verdict=row.get("verdict") or "",
                    time_limit_ms=tl_ms,
                    memory_limit_kb=ml_kb,
                )
                row["exec_time_ms"] = str(exec_ms)
                row["mem_kb"] = str(mem_kb)
                w.writerow(row)

                if args.progress_every > 0 and count % args.progress_every == 0:
                    elapsed = time.time() - start
                    rate = count / elapsed if elapsed > 0 else 0
                    sys.stderr.write(f"\rFilled {count} rows ({rate:.0f}/s)")
                    sys.stderr.flush()

        sys.stderr.write("\n")
        os.replace(tmp, out_path)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
