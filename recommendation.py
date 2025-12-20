"""
recommendation.py

Consolidated training pipeline extracted from:
- 01_derive_students.py
- 02_build_features.py
- 03_train_eval.py
- 04_recommend_eval.py

This script only runs training-related steps and writes model artifacts.
It does not generate any plots or analysis reports.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent

# Inputs (CleanData / FeatureData)
PROBLEMS = ROOT / "CleanData/problems.csv"
SUBMISSIONS = ROOT / "CleanData/submissions.csv"
SUBMISSIONS_COMPAT = ROOT / "CleanData/submissions_clean.csv"
TAGS = ROOT / "CleanData/tags.csv"
LANGS = ROOT / "CleanData/languages.csv"

# Outputs
STUDENTS_OUT = ROOT / "CleanData/students_derived.csv"
TRAIN_SAMPLES_OUT = ROOT / "FeatureData/train_samples.csv"
PIPELINE_OUT = ROOT / "Models/pipeline_logreg.joblib"
RECO_MODEL_OUT = ROOT / "Models/reco_logreg.joblib"

RANDOM_SEED = 42
TIME_SPLIT = 0.8


def parse_json_list(x: object) -> list[str]:
    """
    Parse a CSV cell into a list of strings.

    Supported inputs:
    - Python list
    - JSON list string
    - delimiter-separated strings (comma/semicolon/pipe)
    - None / NaN / empty string
    """
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


def first_existing_path(*candidates: Path) -> Path:
    """Return the first existing path among candidates."""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing input file. Candidates: {[str(c) for c in candidates]!r}")


def derive_students(
    problems_path: Path,
    submissions_path: Path,
    tags_path: Path,
    langs_path: Path,
    out_path: Path,
) -> None:
    problems = pd.read_csv(problems_path)
    subs = pd.read_csv(submissions_path)
    tags = pd.read_csv(tags_path)
    langs = pd.read_csv(langs_path)

    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    problems["tags_norm"] = problems["tags_list"].apply(
        lambda lst: [t for t in lst if t in tag_set][:2]
    )

    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    up = (
        subs.groupby(["user_id", "problem_id"], as_index=False)
        .agg(n_attempts=("submission_id", "count"), solved=("ac", "max"))
        .merge(problems[["problem_id", "difficulty_filled", "tags_norm"]], on="problem_id", how="left")
    )

    up["difficulty_filled"] = up["difficulty_filled"].fillna(diff_median).astype(int)
    up["diff_norm"] = up["difficulty_filled"] / 10.0

    num = (up["solved"] * up["diff_norm"]).groupby(up["user_id"]).sum()
    den = up["diff_norm"].groupby(up["user_id"]).sum()
    level = (num / (den + 1e-9)).reset_index(name="level")

    attempt_stats = (
        up.groupby("user_id")["n_attempts"]
        .mean()
        .reset_index(name="avg_attempts_per_problem")
    )
    p95 = np.percentile(attempt_stats["avg_attempts_per_problem"], 95)
    denom_p = math.log1p(p95) if p95 > 0 else 1.0
    attempt_stats["perseverance"] = attempt_stats["avg_attempts_per_problem"].apply(
        lambda x: min(1.0, math.log1p(x) / denom_p if denom_p > 0 else 0.0)
    )

    known_langs = set(langs["name"].astype(str))
    lang_counts = subs.groupby(["user_id", "language"]).size().reset_index(name="cnt")
    lang_counts = lang_counts[lang_counts["language"].isin(known_langs)]
    lang_tab = lang_counts.pivot_table(
        index="user_id", columns="language", values="cnt", fill_value=0
    )
    lang_tab = lang_tab.div(lang_tab.sum(axis=1).replace(0, 1), axis=0)
    lang_keys = sorted(list(known_langs))
    lang_pref = (
        lang_tab.apply(
            lambda r: json.dumps(
                {k: float(r.get(k, 0.0)) for k in lang_keys if r.get(k, 0.0) > 0},
                ensure_ascii=False,
            ),
            axis=1,
        )
        .reset_index()
    )
    lang_pref.columns = ["user_id", "lang_pref"]

    tag_rows: list[tuple[int, str]] = []
    for uid, tags_norm in up[["user_id", "tags_norm"]].itertuples(index=False):
        if not isinstance(tags_norm, list):
            continue
        for t in tags_norm:
            tag_rows.append((int(uid), str(t)))
    tag_df = pd.DataFrame(tag_rows, columns=["user_id", "tag"])
    tag_counts = tag_df.groupby(["user_id", "tag"]).size().reset_index(name="cnt")
    tag_tab = (
        tag_counts.pivot_table(index="user_id", columns="tag", values="cnt", fill_value=0)
        .reindex(columns=tag_vocab, fill_value=0)
    )
    tag_tab = tag_tab.div(tag_tab.sum(axis=1).replace(0, 1), axis=0)
    tag_pref = (
        tag_tab.apply(
            lambda r: json.dumps(
                {k: float(r.get(k, 0.0)) for k in tag_vocab if r.get(k, 0.0) > 0},
                ensure_ascii=False,
            ),
            axis=1,
        )
        .reset_index()
    )
    tag_pref.columns = ["user_id", "tag_pref"]

    users = pd.DataFrame({"user_id": sorted(subs["user_id"].unique().tolist())})
    out = (
        users.merge(level, on="user_id", how="left")
        .merge(attempt_stats[["user_id", "perseverance"]], on="user_id", how="left")
        .merge(lang_pref, on="user_id", how="left")
        .merge(tag_pref, on="user_id", how="left")
    )
    out["level"] = out["level"].fillna(0).clip(0, 1)
    out["perseverance"] = out["perseverance"].fillna(0).clip(0, 1)
    out["lang_pref"] = out["lang_pref"].fillna("{}")
    out["tag_pref"] = out["tag_pref"].fillna("{}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Wrote", str(out_path), "rows=", len(out))


def build_train_samples(
    problems_path: Path,
    submissions_path: Path,
    tags_path: Path,
    langs_path: Path,
    out_path: Path,
) -> None:
    problems = pd.read_csv(problems_path)
    subs = pd.read_csv(submissions_path, low_memory=False)
    tags = pd.read_csv(tags_path)
    langs = pd.read_csv(langs_path)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Wrote", str(out_path), "rows=", len(out), "cols=", out.shape[1])


def load_train_samples(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing train samples: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"ac", "submission_id", "user_id", "problem_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Train samples missing columns: {sorted(missing)}")

    y = df["ac"].astype(int).to_numpy()
    submission_id = df["submission_id"].to_numpy()

    X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"]).copy()
    feature_cols = list(X.columns)

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X, y, submission_id, feature_cols


def time_split_by_submission_id(
    X: pd.DataFrame,
    y: np.ndarray,
    submission_id: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if len(X) != len(y) or len(X) != len(submission_id):
        raise ValueError("X/y/submission_id length mismatch")

    order = np.argsort(submission_id)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y[order]

    split_idx = int(len(X_sorted) * train_ratio)
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]
    return X_train, X_test, y_train, y_test


def build_candidate_models(random_seed: int) -> dict[str, Any]:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, random_state=random_seed)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=10, random_state=random_seed),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LinearSVC(random_state=random_seed)),
            ]
        ),
    }


def train_baseline_models(X_train: pd.DataFrame, y_train: np.ndarray) -> None:
    models = build_candidate_models(random_seed=RANDOM_SEED)
    for name, model in models.items():
        model.fit(X_train, y_train)
        print("Trained baseline model:", name)


def save_webapp_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    out_path: Path,
    random_seed: int,
) -> None:
    final_logreg = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=random_seed)),
        ]
    )
    final_logreg.fit(X.to_numpy(dtype=np.float32), y.astype(int))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_logreg,
            "feature_cols": feature_cols,
            "random_seed": random_seed,
            "train_rows": int(len(X)),
        },
        out_path,
    )
    print("Saved pipeline to", out_path)


def train_reco_model(
    train_samples_path: Path,
    out_path: Path,
    train_ratio: float,
    random_seed: int,
) -> None:
    df = pd.read_csv(train_samples_path)
    df["submission_id"] = pd.to_numeric(df["submission_id"], errors="coerce").astype(int)
    df = df.sort_values("submission_id").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns if c not in {"ac", "submission_id", "user_id", "problem_id"}
    ]
    if not feature_cols:
        raise ValueError("No feature columns found for recommendation model.")

    split = int(len(df) * train_ratio)
    if split <= 0:
        raise ValueError("Not enough rows for recommendation training split.")
    cutoff_id = int(df["submission_id"].iloc[split - 1])

    train_df = df.iloc[:split].reset_index(drop=True)
    X_train = train_df[feature_cols].copy()
    y_train = train_df["ac"].astype(int).to_numpy()

    for col in X_train.columns:
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(int)

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=random_seed)),
        ]
    )
    model.fit(X_train.to_numpy(dtype=np.float32), y_train.astype(int))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "feature_cols": feature_cols,
            "random_seed": random_seed,
            "train_rows": int(len(train_df)),
            "cutoff_submission_id": cutoff_id,
            "train_ratio": float(train_ratio),
        },
        out_path,
    )
    print("Saved recommendation model to", out_path)


def main() -> int:
    np.random.seed(RANDOM_SEED)

    submissions_path = first_existing_path(SUBMISSIONS, SUBMISSIONS_COMPAT)

    print("Step 1/4: derive student profiles")
    derive_students(
        problems_path=PROBLEMS,
        submissions_path=submissions_path,
        tags_path=TAGS,
        langs_path=LANGS,
        out_path=STUDENTS_OUT,
    )

    print("Step 2/4: build training samples")
    build_train_samples(
        problems_path=PROBLEMS,
        submissions_path=submissions_path,
        tags_path=TAGS,
        langs_path=LANGS,
        out_path=TRAIN_SAMPLES_OUT,
    )

    print("Step 3/4: train baseline models and web pipeline")
    X, y, submission_id, feature_cols = load_train_samples(TRAIN_SAMPLES_OUT)
    X_train, _X_test, y_train, _y_test = time_split_by_submission_id(
        X, y, submission_id, train_ratio=0.8
    )
    train_baseline_models(X_train, y_train)
    save_webapp_pipeline(X=X, y=y, feature_cols=feature_cols, out_path=PIPELINE_OUT, random_seed=RANDOM_SEED)

    print("Step 4/4: train recommendation model")
    train_reco_model(
        train_samples_path=TRAIN_SAMPLES_OUT,
        out_path=RECO_MODEL_OUT,
        train_ratio=TIME_SPLIT,
        random_seed=RANDOM_SEED,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
