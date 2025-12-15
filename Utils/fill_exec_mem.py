import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


TIME_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?")
MEM_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?")


LANGUAGES = ["Python", "C", "C++", "JS", "JAVA", "GO"]
VERDICTS = ["AC", "WA", "TLE", "RE", "CE"]


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
    return max(1, int(round(ms)))


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
    return max(1, int(round(kb)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_int(s: Any) -> int | None:
    try:
        return int(str(s).strip())
    except Exception:
        return None


def parse_json_list_cell(x: Any) -> list[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("["):
        try:
            v = json.loads(s)
        except Exception:
            v = None
        if isinstance(v, list) and all(isinstance(t, str) for t in v):
            return [t.strip() for t in v if t.strip()]
    s = s.strip("[]")
    parts = re.split(r"[;,]\s*|\s+\|\s+|\s+,\s+", s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def weighted_choice(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    cum = 0.0
    for item, w in zip(items, weights, strict=False):
        cum += float(w)
        if r <= cum:
            return item
    return items[-1]


def dirichlet(rng: random.Random, alpha: list[float]) -> list[float]:
    xs = [rng.gammavariate(a, 1.0) if a > 0 else 0.0 for a in alpha]
    s = sum(xs)
    if s <= 0:
        return [1.0 / len(alpha) for _ in alpha]
    return [x / s for x in xs]


@dataclass(frozen=True)
class Problem:
    problem_id: int
    difficulty: int
    tags: list[str]
    time_limit_ms: int
    memory_limit_kb: int


@dataclass
class StudentProfile:
    user_id: int
    level: float
    perseverance: float
    lang_pref: dict[str, float]
    tag_pref: dict[str, float]


def load_tag_vocab(tags_csv: str) -> list[str]:
    with open(tags_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "tag_name" not in r.fieldnames:
            raise SystemExit(f"{tags_csv} missing column 'tag_name'")
        out = []
        for row in r:
            v = (row.get("tag_name") or "").strip()
            if v:
                out.append(v)
        if not out:
            raise SystemExit(f"No tags loaded from {tags_csv}")
        return out


def load_language_vocab(languages_csv: str) -> list[str]:
    with open(languages_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"{languages_csv} missing header")
        col = "name" if "name" in (r.fieldnames or []) else None
        if not col:
            raise SystemExit(f"{languages_csv} missing column 'name'")
        out = []
        for row in r:
            v = (row.get(col) or "").strip()
            if v:
                out.append(v)
        return out


def load_students_user_ids(students_csv: str) -> list[int]:
    with open(students_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "user_id" not in r.fieldnames:
            raise SystemExit(f"{students_csv} missing column 'user_id'")
        ids = []
        for row in r:
            uid = _parse_int(row.get("user_id"))
            if uid is not None:
                ids.append(uid)
        if not ids:
            raise SystemExit(f"No user_id loaded from {students_csv}")
        return ids


def try_load_profiles(
    *,
    students_csv: str,
    tag_vocab: list[str],
    lang_vocab: list[str],
) -> tuple[list[StudentProfile], bool]:
    with open(students_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if "user_id" not in fieldnames:
            raise SystemExit(f"{students_csv} missing column 'user_id'")

        has_all = all(x in fieldnames for x in ("level", "perseverance", "lang_pref", "tag_pref"))
        rows = list(r)

    profiles: list[StudentProfile] = []
    if not has_all:
        for row in rows:
            uid = _parse_int(row.get("user_id"))
            if uid is None:
                continue
            profiles.append(
                StudentProfile(
                    user_id=uid,
                    level=0.0,
                    perseverance=0.0,
                    lang_pref={},
                    tag_pref={},
                )
            )
        return profiles, False

    nontrivial = 0
    for row in rows:
        uid = _parse_int(row.get("user_id"))
        if uid is None:
            continue
        level = float(row.get("level") or 0.0)
        perseverance = float(row.get("perseverance") or 0.0)
        try:
            lang_pref = json.loads(row.get("lang_pref") or "{}")
        except Exception:
            lang_pref = {}
        try:
            tag_pref = json.loads(row.get("tag_pref") or "{}")
        except Exception:
            tag_pref = {}

        # Treat all-zero / all-empty profiles as missing (common in initialized templates).
        if (level > 1e-9) or (perseverance > 1e-9) or bool(lang_pref) or bool(tag_pref):
            nontrivial += 1

        profiles.append(
            StudentProfile(
                user_id=uid,
                level=clamp01(level),
                perseverance=clamp01(perseverance),
                lang_pref={str(k): float(v) for k, v in (lang_pref or {}).items()},
                tag_pref={str(k): float(v) for k, v in (tag_pref or {}).items()},
            )
        )

    # If almost all profiles are trivial, regenerate.
    if profiles and nontrivial / float(len(profiles)) < 0.10:
        return profiles, False
    return profiles, True


def generate_profiles(
    rng: random.Random,
    user_ids: list[int],
    *,
    tag_vocab: list[str],
    lang_vocab: list[str],
) -> list[StudentProfile]:
    profiles: list[StudentProfile] = []
    for uid in user_ids:
        # level: mixture beta -> mean ~0.5 with tails
        roll = rng.random()
        if roll < 0.15:
            level = rng.betavariate(0.9, 3.5)
        elif roll < 0.30:
            level = rng.betavariate(3.5, 0.9)
        else:
            level = rng.betavariate(2.2, 2.2)

        # perseverance: moderate with variance (not all near 1)
        p_roll = rng.random()
        if p_roll < 0.15:
            perseverance = rng.betavariate(1.5, 3.0)
        else:
            perseverance = rng.betavariate(2.8, 1.8)

        # lang/tag prefs: peaked Dirichlet
        lang_alpha = [0.35 for _ in lang_vocab]
        tag_alpha = [0.45 for _ in tag_vocab]
        lang_p = dirichlet(rng, lang_alpha)
        tag_p = dirichlet(rng, tag_alpha)
        lang_pref = {k: float(v) for k, v in zip(lang_vocab, lang_p, strict=False)}
        tag_pref = {k: float(v) for k, v in zip(tag_vocab, tag_p, strict=False)}

        profiles.append(
            StudentProfile(
                user_id=int(uid),
                level=clamp01(float(level)),
                perseverance=clamp01(float(perseverance)),
                lang_pref=lang_pref,
                tag_pref=tag_pref,
            )
        )
    return profiles


def write_students_derived(
    path: str,
    profiles: list[StudentProfile],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "level", "perseverance", "lang_pref", "tag_pref"])
        for p in profiles:
            w.writerow(
                [
                    p.user_id,
                    f"{p.level:.6f}",
                    f"{p.perseverance:.6f}",
                    json.dumps(p.lang_pref, ensure_ascii=False),
                    json.dumps(p.tag_pref, ensure_ascii=False),
                ]
            )
    os.replace(tmp, path)


def load_problems(
    *,
    problems_csv: str,
    allowed_tags: set[str],
) -> list[Problem]:
    out: list[Problem] = []
    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if "problem_id" not in fieldnames:
            raise SystemExit(f"{problems_csv} missing column 'problem_id'")
        if "difficulty" not in fieldnames:
            raise SystemExit(f"{problems_csv} missing column 'difficulty'")

        for idx, row in enumerate(r, start=1):
            pid = _parse_int(row.get("problem_id")) or idx
            d = _parse_int(row.get("difficulty")) or 5
            d = max(1, min(10, d))
            tags = parse_json_list_cell(row.get("tags"))
            tags = [t for t in tags if t in allowed_tags][:2]
            tl = parse_time_limit_ms(row.get("time_limit") or "")
            ml = parse_memory_limit_kb(row.get("memory_limit") or "")
            out.append(Problem(problem_id=pid, difficulty=d, tags=tags, time_limit_ms=tl, memory_limit_kb=ml))
    if not out:
        raise SystemExit(f"No problems loaded from {problems_csv}")
    return out


def build_problems_by_tag(problems: list[Problem], tag_vocab: list[str]) -> dict[str, list[int]]:
    by: dict[str, list[int]] = {t: [] for t in tag_vocab}
    for p in problems:
        for t in p.tags:
            if t in by:
                by[t].append(p.problem_id)
    return by


def tag_match_score(profile: StudentProfile, problem: Problem) -> float:
    if not problem.tags:
        return 0.0
    vals = [float(profile.tag_pref.get(t, 0.0)) for t in problem.tags]
    return float(sum(vals)) / float(len(vals))


def lang_match_score(profile: StudentProfile, language: str) -> float:
    return float(profile.lang_pref.get(language, 0.0))


def sample_language(rng: random.Random, profile: StudentProfile, lang_vocab: list[str]) -> str:
    # Mostly follow preference, sometimes explore.
    if rng.random() < 0.10:
        return rng.choice(lang_vocab)
    weights = [float(profile.lang_pref.get(l, 0.0)) for l in lang_vocab]
    return weighted_choice(rng, lang_vocab, weights)


def pick_problem_for_user(
    rng: random.Random,
    profile: StudentProfile,
    *,
    all_problem_ids: list[int],
    problems_by_tag: dict[str, list[int]],
    already_attempted: set[int],
    explore_rate: float,
    tag_vocab: list[str],
) -> int | None:
    for _ in range(50):
        if rng.random() < explore_rate:
            pid = rng.choice(all_problem_ids)
        else:
            tag_weights = [float(profile.tag_pref.get(t, 0.0)) for t in tag_vocab]
            tag = weighted_choice(rng, tag_vocab, tag_weights)
            pool = problems_by_tag.get(tag) or []
            pid = rng.choice(pool) if pool else rng.choice(all_problem_ids)
        if pid not in already_attempted:
            return int(pid)
    return None


def verdict_for_failure(
    rng: random.Random,
    *,
    diff: float,
    lang_match: float,
) -> str:
    # Base: WA 0.70, TLE 0.15, RE 0.10, CE 0.05 (with mild adjustments)
    wa = 0.70
    tle = 0.15 + (0.05 if diff >= 0.8 else 0.0) + (0.03 if lang_match <= 0.08 else 0.0)
    re = 0.10
    ce = 0.05 + (0.03 if lang_match <= 0.05 else 0.0)
    weights = [wa, tle, re, ce]
    v = weighted_choice(rng, ["WA", "TLE", "RE", "CE"], weights)
    return v


def generate_exec_mem(
    rng: random.Random,
    *,
    verdict: str,
    time_limit_ms: int,
    memory_limit_kb: int,
    diff: float,
) -> tuple[int, int]:
    tl = max(1, int(time_limit_ms))
    ml = max(1, int(memory_limit_kb))

    if verdict == "CE":
        return 0, 0

    # baseline memory proportional to difficulty, but bounded.
    base_mem = int(ml * (0.10 + 0.55 * diff))
    mem_kb = max(10_000, min(200_000, int(base_mem * rng.uniform(0.6, 1.4))))

    if verdict == "TLE":
        exec_ms = int(tl * rng.uniform(1.05, 3.0))
        return exec_ms, mem_kb

    if verdict == "RE":
        exec_ms = int(tl * rng.uniform(0.05, 0.60))
        return exec_ms, mem_kb

    if verdict == "WA":
        exec_ms = int(tl * rng.uniform(0.10, 0.95))
        return exec_ms, mem_kb

    # AC: usually faster and within limits.
    exec_ms = int(tl * rng.uniform(0.05, 0.70))
    return exec_ms, mem_kb


def generate_submissions(
    *,
    rng: random.Random,
    problems: list[Problem],
    profiles: list[StudentProfile],
    lang_vocab: list[str],
    tag_vocab: list[str],
    output_csv: str,
    target_rows: int,
    explore_rate: float,
    max_attempts: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    bias: float,
    noise: float,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    tmp = output_csv + ".tmp"

    problems_by_id = {p.problem_id: p for p in problems}
    problems_by_tag = build_problems_by_tag(problems, tag_vocab)
    all_problem_ids = [p.problem_id for p in problems]

    # User activity weights to create long tail.
    uids = [p.user_id for p in profiles]
    uweight = []
    profile_by_uid: dict[int, StudentProfile] = {}
    for p in profiles:
        profile_by_uid[p.user_id] = p
        w = 0.25 + 0.90 * p.perseverance + 0.50 * p.level
        uweight.append(max(0.01, float(w)))

    attempted: dict[int, set[int]] = {uid: set() for uid in uids}
    attempt_next: dict[tuple[int, int], int] = {}

    # stats
    total = 0
    ac_total = 0
    by_diff = Counter()
    by_diff_ac = Counter()
    by_attempt = Counter()
    by_lang = Counter()
    by_lang_ac = Counter()
    by_tag = Counter()
    by_tag_ac = Counter()

    start_t = time.time()
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "submission_id",
                "user_id",
                "problem_id",
                "attempt_no",
                "language",
                "verdict",
                "ac",
                "exec_time_ms",
                "mem_kb",
            ]
        )

        submission_id = 0
        while submission_id < target_rows:
            uid = int(weighted_choice(rng, uids, uweight))
            profile = profile_by_uid[uid]
            pid = pick_problem_for_user(
                rng,
                profile,
                all_problem_ids=all_problem_ids,
                problems_by_tag=problems_by_tag,
                already_attempted=attempted[uid],
                explore_rate=explore_rate,
                tag_vocab=tag_vocab,
            )
            if pid is None:
                continue
            attempted[uid].add(pid)

            problem = problems_by_id.get(pid)
            if not problem:
                continue

            k = 1
            while k <= max_attempts and submission_id < target_rows:
                language = sample_language(rng, profile, lang_vocab)

                diff = float(problem.difficulty) / 10.0
                tmatch = tag_match_score(profile, problem)
                lmatch = lang_match_score(profile, language)
                eps = rng.uniform(-noise, noise) if noise > 0 else 0.0
                p_ac = sigmoid(a * profile.level - b * diff + c * tmatch + d * lmatch + e * (k - 1) + bias + eps)
                ac = 1 if rng.random() < p_ac else 0
                verdict = "AC" if ac == 1 else verdict_for_failure(rng, diff=diff, lang_match=lmatch)

                exec_ms, mem_kb = generate_exec_mem(
                    rng,
                    verdict=verdict,
                    time_limit_ms=problem.time_limit_ms,
                    memory_limit_kb=problem.memory_limit_kb,
                    diff=diff,
                )

                submission_id += 1
                attempt_no = attempt_next.get((uid, pid), 1)
                attempt_next[(uid, pid)] = attempt_no + 1

                w.writerow([submission_id, uid, pid, attempt_no, language, verdict, ac, exec_ms, mem_kb])

                # stats
                total += 1
                ac_total += ac
                by_diff[problem.difficulty] += 1
                by_diff_ac[problem.difficulty] += ac
                by_attempt[min(attempt_no, max_attempts)] += 1
                by_lang[language] += 1
                by_lang_ac[language] += ac
                for t in problem.tags:
                    by_tag[t] += 1
                    by_tag_ac[t] += ac

                if ac == 1:
                    break

                continue_prob = profile.perseverance * max(0.0, 1.0 - 0.15 * (k - 1))
                if rng.random() >= continue_prob:
                    break
                k += 1

            if submission_id % 50_000 == 0 and submission_id > 0:
                dt = time.time() - start_t
                print(f"progress rows={submission_id}/{target_rows} elapsed={dt:.1f}s")

    os.replace(tmp, output_csv)

    # Log summary stats (not written to CSV)
    def safe_rate(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    print(f"OK wrote: {output_csv} rows={total}")
    print(f"overall AC rate: {safe_rate(ac_total, total):.4f}")
    print("AC rate by difficulty(1-10):")
    for d0 in range(1, 11):
        print(f"  diff={d0}: {safe_rate(by_diff_ac[d0], by_diff[d0]):.4f} (n={by_diff[d0]})")
    print("attempt_no distribution (1..10):")
    for k0 in range(1, max_attempts + 1):
        print(f"  attempt_no={k0}: {by_attempt[k0]}")
    print("AC rate by language:")
    for lang in lang_vocab:
        print(f"  {lang}: {safe_rate(by_lang_ac[lang], by_lang[lang]):.4f} (n={by_lang[lang]})")
    print("AC rate by tag:")
    for tag in tag_vocab:
        print(f"  {tag}: {safe_rate(by_tag_ac[tag], by_tag[tag]):.4f} (n={by_tag[tag]})")


def fill_exec_mem_inplace_or_copy(
    *,
    rng: random.Random,
    problems_csv: str,
    submissions_csv: str,
    output_csv: str,
    inplace: bool,
) -> None:
    limits: dict[int, tuple[int, int]] = {}
    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for idx, row in enumerate(r, start=1):
            pid_raw = (row.get("problem_id") or "").strip()
            pid = int(pid_raw) if pid_raw.isdigit() else idx
            tl_ms = parse_time_limit_ms(row.get("time_limit") or "")
            ml_kb = parse_memory_limit_kb(row.get("memory_limit") or "")
            limits[pid] = (tl_ms, ml_kb)

    out_path = submissions_csv if inplace else output_csv
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(submissions_csv, "r", encoding="utf-8-sig", newline="") as f_in:
        r = csv.DictReader(f_in)
        if not r.fieldnames:
            raise SystemExit("submissions.csv missing header")
        fieldnames = list(r.fieldnames)
        if "exec_time_ms" not in fieldnames or "mem_kb" not in fieldnames:
            raise SystemExit("submissions.csv must contain exec_time_ms and mem_kb columns")
        if "problem_id" not in fieldnames or "verdict" not in fieldnames:
            raise SystemExit("submissions.csv must contain problem_id and verdict columns")

        with open(tmp, "w", encoding="utf-8-sig", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in r:
                pid = _parse_int(row.get("problem_id")) or 0
                tl_ms, ml_kb = limits.get(pid, (1000, 128 * 1024))
                verdict = (row.get("verdict") or "").strip().upper()
                diff = 0.5
                exec_ms, mem_kb = generate_exec_mem(
                    rng,
                    verdict=verdict,
                    time_limit_ms=tl_ms,
                    memory_limit_kb=ml_kb,
                    diff=diff,
                )
                row["exec_time_ms"] = str(exec_ms)
                row["mem_kb"] = str(mem_kb)
                w.writerow(row)

    os.replace(tmp, out_path)
    print(f"OK wrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a correlated OJ submissions dataset (recommended), or fill exec_time_ms/mem_kb for an existing file."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    g = sub.add_parser("generate", help="Generate submissions.csv with multi-factor correlations")
    g.add_argument("--problems", default="CleanData/problems.csv")
    g.add_argument("--students", default="CleanData/students.csv")
    g.add_argument("--languages", default="CleanData/languages.csv")
    g.add_argument("--tags", default="CleanData/tags.csv")
    g.add_argument("--output", default="CleanData/submissions.csv")
    g.add_argument("--target-rows", type=int, default=300_000)
    g.add_argument("--explore-rate", type=float, default=0.20)
    g.add_argument("--max-attempts", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--write-students-derived", default="", help="Write generated profiles CSV (if needed)")

    # Sigmoid params
    g.add_argument("--a", type=float, default=3.0)
    g.add_argument("--b", type=float, default=4.0)
    g.add_argument("--c", type=float, default=0.6)
    g.add_argument("--d", type=float, default=0.3)
    g.add_argument("--e", type=float, default=0.4)
    g.add_argument("--bias", type=float, default=0.0)
    g.add_argument("--noise", type=float, default=0.05)

    f = sub.add_parser("fill", help="Fill exec_time_ms/mem_kb in an existing submissions CSV")
    f.add_argument("--problems", default="CleanData/problems.csv")
    f.add_argument("--submissions", default="CleanData/submissions.csv")
    f.add_argument("--output", default="CleanData/submissions_filled.csv")
    f.add_argument("--inplace", action="store_true")
    f.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    cmd = args.cmd or "fill"

    rng = random.Random(int(getattr(args, "seed", 42)))

    if cmd == "fill":
        fill_exec_mem_inplace_or_copy(
            rng=rng,
            problems_csv=args.problems,
            submissions_csv=args.submissions,
            output_csv=args.output,
            inplace=bool(args.inplace),
        )
        return 0

    if cmd != "generate":
        raise SystemExit(f"Unknown cmd: {cmd}")

    tag_vocab = load_tag_vocab(args.tags)
    lang_vocab = load_language_vocab(args.languages) or LANGUAGES
    # Enforce fixed language set if present; fall back to fixed.
    lang_vocab = [l for l in LANGUAGES if l in set(lang_vocab)] or LANGUAGES

    user_ids = load_students_user_ids(args.students)
    profiles, has_profiles = try_load_profiles(students_csv=args.students, tag_vocab=tag_vocab, lang_vocab=lang_vocab)
    if not has_profiles:
        profiles = generate_profiles(rng, user_ids, tag_vocab=tag_vocab, lang_vocab=lang_vocab)
        out_students = (args.write_students_derived or "").strip()
        if out_students:
            write_students_derived(out_students, profiles)

    problems = load_problems(problems_csv=args.problems, allowed_tags=set(tag_vocab))

    generate_submissions(
        rng=rng,
        problems=problems,
        profiles=profiles,
        lang_vocab=lang_vocab,
        tag_vocab=tag_vocab,
        output_csv=args.output,
        target_rows=max(1, int(args.target_rows)),
        explore_rate=clamp01(float(args.explore_rate)),
        max_attempts=max(1, min(50, int(args.max_attempts))),
        a=float(args.a),
        b=float(args.b),
        c=float(args.c),
        d=float(args.d),
        e=float(args.e),
        bias=float(args.bias),
        noise=max(0.0, float(args.noise)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
