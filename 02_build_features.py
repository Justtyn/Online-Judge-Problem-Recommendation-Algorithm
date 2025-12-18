import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

PROBLEMS = ROOT / "CleanData/problems.csv"
SUBS = ROOT / "CleanData/submissions.csv"
SUBS_COMPAT = ROOT / "CleanData/submissions_clean.csv"
TAGS = ROOT / "CleanData/tags.csv"
LANGS = ROOT / "CleanData/languages.csv"
OUT = ROOT / "FeatureData/train_samples.csv"
RANDOM_SEED = 42


def first_existing_path(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到输入文件，候选路径：{[str(c) for c in candidates]!r}")


def parse_json_list(x: object) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass
    s = s.strip("[]")
    parts = re.split(r"[;,]\s*|\s+\|\s+|\s+,\s+", s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def main() -> int:
    np.random.seed(RANDOM_SEED)

    problems = pd.read_csv(PROBLEMS)
    subs = pd.read_csv(first_existing_path(SUBS, SUBS_COMPAT), low_memory=False)
    tags = pd.read_csv(TAGS)
    langs = pd.read_csv(LANGS)

    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    lang_vocab = sorted(set(langs["name"].astype(str).tolist()))
    lang_set = set(lang_vocab)

    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["diff_norm"] = problems["difficulty_filled"].astype(float) / 10.0
    problems["tags_list"] = problems["tags"].apply(parse_json_list)
    problems["tags2"] = problems["tags_list"].apply(lambda lst: [t for t in lst if t in tag_set][:2])

    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int)))
    pid_to_diffnorm = dict(zip(problems["problem_id"].astype(int), problems["diff_norm"].astype(float)))
    pid_to_tags2 = dict(zip(problems["problem_id"].astype(int), problems["tags2"].tolist()))

    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)
    subs["language"] = subs.get("language", "").astype(str).fillna("")

    subs = subs.sort_values(["user_id", "submission_id"]).reset_index(drop=True)

    user_total = subs.groupby("user_id").size()
    user_unique_prob = subs.groupby("user_id")["problem_id"].nunique()
    avg_attempts = (user_total / user_unique_prob.replace(0, np.nan)).dropna().values.astype(float)
    p95 = float(np.percentile(avg_attempts, 95)) if len(avg_attempts) else 1.0
    denom_p = math.log1p(p95) if p95 > 0 else 1.0

    n = len(subs)
    level_arr = np.zeros((n,), dtype=np.float32)
    pers_arr = np.zeros((n,), dtype=np.float32)
    lang_match_arr = np.zeros((n,), dtype=np.float32)
    tag_match_arr = np.zeros((n,), dtype=np.float32)

    tags2_list = subs["problem_id"].map(lambda pid: pid_to_tags2.get(int(pid), [])).to_list()
    diff_filled = subs["problem_id"].map(lambda pid: pid_to_diff.get(int(pid), diff_median)).astype(int)
    diff_norm = subs["problem_id"].map(lambda pid: pid_to_diffnorm.get(int(pid), diff_median / 10.0)).astype(float)

    tag_to_j = {t: j for j, t in enumerate(tag_vocab)}

    for uid, idx in subs.groupby("user_id", sort=False).groups.items():
        idx_list = idx.tolist()
        problems_attempted: set[int] = set()
        problems_solved: set[int] = set()
        denom_sum = 0.0
        num_sum = 0.0
        total_subs = 0
        lang_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        total_tag = 0

        for i in idx_list:
            pid = int(subs.at[i, "problem_id"])
            lang = str(subs.at[i, "language"] or "")
            ac = int(subs.at[i, "ac"])

            if denom_sum > 0:
                level_arr[i] = float(num_sum / (denom_sum + 1e-9))
            else:
                level_arr[i] = 0.0

            if total_subs > 0 and len(problems_attempted) > 0 and denom_p > 0:
                avg = float(total_subs) / float(len(problems_attempted))
                pers_arr[i] = float(max(0.0, min(1.0, math.log1p(avg) / denom_p)))
            else:
                pers_arr[i] = 0.0

            if total_subs > 0 and lang in lang_set:
                lang_match_arr[i] = float(lang_counts.get(lang, 0) / total_subs)
            else:
                lang_match_arr[i] = 0.0

            t2 = tags2_list[i]
            if t2 and total_tag > 0:
                tag_match_arr[i] = float(sum(tag_counts.get(t, 0) / total_tag for t in t2) / len(t2))
            else:
                tag_match_arr[i] = 0.0

            total_subs += 1
            if lang in lang_set:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if pid not in problems_attempted:
                problems_attempted.add(pid)
                denom_sum += float(diff_norm.iat[i])
                for t in t2:
                    if t in tag_to_j:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                        total_tag += 1

            if ac == 1 and pid not in problems_solved:
                problems_solved.add(pid)
                num_sum += float(diff_norm.iat[i])

    lang_ohe = pd.get_dummies(subs["language"], prefix="lang")
    for l in lang_vocab:
        c = f"lang_{l}"
        if c not in lang_ohe.columns:
            lang_ohe[c] = 0
    lang_ohe = lang_ohe[[f"lang_{l}" for l in lang_vocab]]

    tag_mh = np.zeros((n, len(tag_vocab)), dtype=np.uint8)
    for i, t2 in enumerate(tags2_list):
        if not t2:
            continue
        for t in t2:
            j = tag_to_j.get(t)
            if j is not None:
                tag_mh[i, j] = 1
    tag_mh_df = pd.DataFrame(tag_mh, columns=[f"tag_{t}" for t in tag_vocab])

    out = pd.concat(
        [
            subs[["submission_id", "user_id", "problem_id", "attempt_no"]].astype(int),
            pd.DataFrame(
                {
                    "difficulty_filled": diff_filled.astype(int),
                    "level": level_arr.astype(float),
                    "perseverance": pers_arr.astype(float),
                    "lang_match": lang_match_arr.astype(float),
                    "tag_match": tag_match_arr.astype(float),
                }
            ),
            lang_ohe.astype(int),
            tag_mh_df.astype(int),
            subs[["ac"]].astype(int),
        ],
        axis=1,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("Wrote", str(OUT), "rows=", len(out), "cols=", out.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
