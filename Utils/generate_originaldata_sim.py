import argparse
import csv
import json
import math
import os
import random
import sys
import time


LANGUAGES = ["Python", "C", "C++", "JS", "JAVA", "GO"]
VERDICTS_FAIL = ["WA", "TLE", "RE", "CE"]
VERDICTS_ALL = ["AC", "WA", "TLE", "RE", "CE"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def read_problem_difficulties(problems_csv: str) -> list[float]:
    diffs: list[float] = []
    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = (row.get("difficulty") or "").strip()
            if v.isdigit():
                d = int(v)
                d = max(1, min(10, d))
                diffs.append((d - 1) / 9.0)
            else:
                diffs.append(0.5)
    return diffs


def add_problem_ids_inplace(problems_csv: str, *, dry_run: bool) -> int:
    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"Empty header: {problems_csv}")
        fieldnames = list(r.fieldnames)
        rows = list(r)

    if "problem_id" not in fieldnames:
        fieldnames = ["problem_id"] + fieldnames

    for idx, row in enumerate(rows, start=1):
        row["problem_id"] = str(idx)

    if dry_run:
        return len(rows)

    tmp = problems_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, problems_csv)
    return len(rows)


def write_students_csv(students_csv: str, *, n_students: int, dry_run: bool) -> None:
    ensure_parent_dir(students_csv)
    if dry_run:
        return
    with open(students_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "level", "perseverance", "lang_pref", "tag_pref"])
        for user_id in range(1, n_students + 1):
            w.writerow([user_id, 0, 0, "{}", "{}"])


def write_languages_csv(languages_csv: str, *, dry_run: bool) -> None:
    ensure_parent_dir(languages_csv)
    if dry_run:
        return
    rows = [
        (1, "Python", "Python 3"),
        (2, "C", "C (GCC)"),
        (3, "C++", "C++ (G++)"),
        (4, "JS", "JavaScript (Node.js)"),
        (5, "JAVA", "Java"),
        (6, "GO", "Go"),
    ]
    with open(languages_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language_id", "name", "desc"])
        for r in rows:
            w.writerow(list(r))


def write_verdicts_csv(verdicts_csv: str, *, dry_run: bool) -> None:
    ensure_parent_dir(verdicts_csv)
    if dry_run:
        return
    rows = [
        (1, "AC", "Accepted"),
        (2, "WA", "Wrong Answer"),
        (3, "TLE", "Time Limit Exceeded"),
        (4, "RE", "Runtime Error"),
        (5, "CE", "Compilation Error"),
    ]
    with open(verdicts_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["verdict_id", "name", "desc"])
        for r in rows:
            w.writerow(list(r))


def choose_weighted(rng: random.Random, items: list[str], weights: list[float]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def generate_submissions_csv(
    submissions_csv: str,
    *,
    n_submissions: int,
    n_students: int,
    n_problems: int,
    problem_difficulty01: list[float],
    rng: random.Random,
    dry_run: bool,
) -> None:
    ensure_parent_dir(submissions_csv)

    user_ability = [rng.betavariate(2.0, 2.0) for _ in range(n_students + 1)]
    user_perseverance = [rng.betavariate(2.0, 2.0) for _ in range(n_students + 1)]
    user_lang_weights = [
        None
    ] + [[rng.random() + 0.1 for _ in LANGUAGES] for _ in range(n_students)]

    attempt_no: dict[tuple[int, int], int] = {}
    solved: set[tuple[int, int]] = set()

    if dry_run:
        return

    with open(submissions_csv, "w", encoding="utf-8-sig", newline="") as f:
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

        start_time = time.time()
        for submission_id in range(1, n_submissions + 1):
            if submission_id % 5000 == 0:
                elapsed = time.time() - start_time
                rate = submission_id / elapsed if elapsed > 0 else 0
                sys.stderr.write(
                    f"\rGenerated {submission_id}/{n_submissions} submissions ({rate:.0f}/s)"
                )
                sys.stderr.flush()

            user_id = rng.randint(1, n_students)

            # Prefer continuing on existing unsolved problems based on perseverance.
            if rng.random() < user_perseverance[user_id] and attempt_no:
                # Try a few times to find an unsolved existing pair for this user.
                picked_problem_id = 0
                for _ in range(5):
                    pid = rng.randint(1, n_problems)
                    if (user_id, pid) in attempt_no and (user_id, pid) not in solved:
                        picked_problem_id = pid
                        break
                if not picked_problem_id:
                    picked_problem_id = rng.randint(1, n_problems)
            else:
                picked_problem_id = rng.randint(1, n_problems)

            # Avoid generating more submissions after AC for the same (user, problem).
            for _ in range(1000):
                if (user_id, picked_problem_id) not in solved:
                    break
                picked_problem_id = rng.randint(1, n_problems)
            else:
                raise RuntimeError(
                    f"Unable to find an unsolved problem for user_id={user_id}; increase n_problems or adjust simulation."
                )

            key = (user_id, picked_problem_id)
            attempt = attempt_no.get(key, 0) + 1
            attempt_no[key] = attempt

            language = choose_weighted(rng, LANGUAGES, user_lang_weights[user_id])

            diff01 = problem_difficulty01[picked_problem_id - 1]
            ability = user_ability[user_id]
            base = (ability - diff01) * 4.0
            learn = (attempt - 1) * 0.7
            p_ac = sigmoid(base + learn)

            if rng.random() < p_ac:
                verdict = "AC"
                ac = 1
                solved.add(key)
            else:
                verdict = choose_weighted(rng, VERDICTS_FAIL, [0.7, 0.1, 0.15, 0.05])
                ac = 0

            w.writerow(
                [
                    submission_id,
                    user_id,
                    picked_problem_id,
                    attempt,
                    language,
                    verdict,
                    ac,
                    0,
                    0,
                ]
            )

        sys.stderr.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate simulated students/submissions CSVs under OriginalData and add problem_id to problems.csv."
    )
    parser.add_argument(
        "--problems-csv",
        default="OriginalData/problems.csv",
        help="Problems CSV (will be updated in-place to add problem_id only; title is unchanged)",
    )
    parser.add_argument(
        "--students-csv",
        default="OriginalData/students.csv",
        help="Output students.csv (placeholder columns filled with 0/{} as suggested)",
    )
    parser.add_argument(
        "--languages-csv",
        default="OriginalData/languages.csv",
        help="Output languages.csv (id,name,desc lookup table)",
    )
    parser.add_argument(
        "--verdicts-csv",
        default="OriginalData/verdicts.csv",
        help="Output verdicts.csv (id,name,desc lookup table)",
    )
    parser.add_argument(
        "--submissions-csv",
        default="OriginalData/submissions.csv",
        help="Output submissions.csv (simulated behavior logs)",
    )
    parser.add_argument("--n-students", type=int, default=1000, help="Number of students")
    parser.add_argument(
        "--n-submissions", type=int, default=100000, help="Number of submissions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write any files; only validate/plan"
    )
    args = parser.parse_args()

    if args.n_students <= 0:
        raise SystemExit("--n-students must be > 0")
    if args.n_submissions <= 0:
        raise SystemExit("--n-submissions must be > 0")

    rng = random.Random(args.seed)

    n_problems = add_problem_ids_inplace(args.problems_csv, dry_run=args.dry_run)
    problem_difficulty01 = read_problem_difficulties(args.problems_csv)
    if len(problem_difficulty01) != n_problems:
        raise SystemExit("Internal error: difficulty list length mismatch.")

    write_languages_csv(args.languages_csv, dry_run=args.dry_run)
    write_verdicts_csv(args.verdicts_csv, dry_run=args.dry_run)
    write_students_csv(args.students_csv, n_students=args.n_students, dry_run=args.dry_run)
    generate_submissions_csv(
        args.submissions_csv,
        n_submissions=args.n_submissions,
        n_students=args.n_students,
        n_problems=n_problems,
        problem_difficulty01=problem_difficulty01,
        rng=rng,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "problems_csv": args.problems_csv,
                    "problems_count": n_problems,
                    "students_csv": args.students_csv,
                    "students_count": args.n_students,
                    "languages_csv": args.languages_csv,
                    "verdicts_csv": args.verdicts_csv,
                    "submissions_csv": args.submissions_csv,
                    "submissions_count": args.n_submissions,
                    "seed": args.seed,
                },
                ensure_ascii=False,
            )
        )
    else:
        print("OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
