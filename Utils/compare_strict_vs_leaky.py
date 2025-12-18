"""
Utils/compare_strict_vs_leaky.py

用途
- 对比“严格无泄漏（strict）”与“存在时间泄漏（leaky）”两种特征口径在：
  - 特征分布（均值/中位数/P95/标准差）
  - 分类模型效果（Accuracy/Precision/Recall/F1/ROC-AUC/Brier 等）
  - 推荐离线评估（Hit@K/Precision@K 等）
  上的差异，帮助解释“为何随机/泄漏口径会虚高指标”。

输入（默认）
- strict 数据：`FeatureData/train_samples.csv`
- submissions：`CleanData/submissions.csv`
- problems：`CleanData/problems.csv`
- students_derived：`CleanData/students_derived.csv`
- tags / languages：`CleanData/tags.csv` / `CleanData/languages.csv`

输出（默认）
- 表格/报告：`Reports/compare/compare_*.csv` 与 `Reports/compare/*.md`
- 图表：`Reports/fig/fig_compare_*.png`（ROC/PR/校准曲线、Hit@K 曲线等）

说明
- 该脚本需要 `matplotlib` 与 `scikit-learn`；并强制使用 Agg 后端以支持无 GUI 环境运行。
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def parse_json_list(x: object) -> list[str]:
    """
    将“可能是 JSON 字符串/分隔字符串/None”的单元格解析为字符串列表。

    主要用于 problems.tags 等字段的兼容解析。
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


def parse_json_dict(x: object) -> dict[str, float]:
    """
    将 “JSON 字符串 / dict / 空值” 解析为 {str: float}。

    主要用于 students_derived 的 lang_pref/tag_pref 等字段。
    """
    if x is None:
        return {}
    if isinstance(x, float) and np.isnan(x):
        return {}
    if isinstance(x, dict):
        out: dict[str, float] = {}
        for k, v in x.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        v = json.loads(s)
    except Exception:
        return {}
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, val in v.items():
        try:
            out[str(k)] = float(val)
        except Exception:
            continue
    return out


def stat_series(s: pd.Series) -> dict[str, float]:
    """对数值序列计算 mean/p50/p95/std（自动丢弃 NaN/inf）。"""
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "std": float("nan")}
    return {
        "mean": float(x.mean()),
        "p50": float(x.median()),
        "p95": float(np.percentile(x.to_numpy(dtype=float), 95)),
        "std": float(x.std(ddof=0)),
    }


@dataclass(frozen=True)
class Split:
    cutoff_submission_id: int
    train_mask: np.ndarray
    test_mask: np.ndarray


def time_split_by_submission_id(df: pd.DataFrame, *, frac: float) -> Split:
    """按 submission_id 升序做时间切分（前 frac 为训练窗，后 1-frac 为测试窗）。"""
    order = np.argsort(df["submission_id"].to_numpy(dtype=np.int64))
    split = int(len(order) * float(frac))
    split = max(1, min(len(order) - 1, split))
    cutoff = int(df["submission_id"].to_numpy(dtype=np.int64)[order[split - 1]])
    train_mask = df["submission_id"].to_numpy(dtype=np.int64) <= cutoff
    test_mask = ~train_mask
    return Split(cutoff_submission_id=cutoff, train_mask=train_mask, test_mask=test_mask)


def build_leaky_dataset(
        *,
        submissions_csv: str,
        problems_csv: str,
        students_derived_csv: str,
        tags_csv: str,
        languages_csv: str,
) -> pd.DataFrame:
    """
    构造“泄漏口径”的训练样本。

    特征口径
    - 用户画像/偏好基于全量 submissions 统计（等价于“看到了未来”），用于与 strict 口径对比。
    """
    subs = pd.read_csv(submissions_csv, low_memory=False)
    problems = pd.read_csv(problems_csv, low_memory=False)
    students = pd.read_csv(students_derived_csv, low_memory=False)
    tags = pd.read_csv(tags_csv, low_memory=False)
    langs = pd.read_csv(languages_csv, low_memory=False)

    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    lang_vocab = sorted(set(langs["name"].astype(str).tolist()))

    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_norm"] = problems["tags"].apply(parse_json_list)
    problems["tags_norm"] = problems["tags_norm"].apply(lambda lst: [t for t in lst if t in tag_set][:2])
    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int)))
    pid_to_tags2 = dict(zip(problems["problem_id"].astype(int), problems["tags_norm"].tolist()))

    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)
    subs["language"] = subs.get("language", "").astype(str).fillna("")

    students["user_id"] = pd.to_numeric(students["user_id"], errors="coerce").astype(int)
    students["level"] = pd.to_numeric(students["level"], errors="coerce").fillna(0.0).astype(float)
    students["perseverance"] = pd.to_numeric(students["perseverance"], errors="coerce").fillna(0.0).astype(float)
    students["lang_pref_dict"] = students["lang_pref"].apply(parse_json_dict)
    students["tag_pref_dict"] = students["tag_pref"].apply(parse_json_dict)

    user_level = students.set_index("user_id")["level"].to_dict()
    user_pers = students.set_index("user_id")["perseverance"].to_dict()
    user_lang_pref = students.set_index("user_id")["lang_pref_dict"].to_dict()
    user_tag_pref = students.set_index("user_id")["tag_pref_dict"].to_dict()

    def lang_match_for_row(uid: int, language: str) -> float:
        pref = user_lang_pref.get(int(uid), {}) or {}
        try:
            return float(pref.get(str(language), 0.0))
        except Exception:
            return 0.0

    def tag_match_for_row(uid: int, tags2: list[str]) -> float:
        pref = user_tag_pref.get(int(uid), {}) or {}
        if not tags2:
            return 0.0
        vals = []
        for t in tags2:
            try:
                vals.append(float(pref.get(str(t), 0.0)))
            except Exception:
                vals.append(0.0)
        return float(sum(vals) / len(vals)) if vals else 0.0

    tags2_list = subs["problem_id"].map(lambda pid: pid_to_tags2.get(int(pid), [])).to_list()
    diff_filled = subs["problem_id"].map(lambda pid: pid_to_diff.get(int(pid), diff_median)).astype(int)

    lang_match = np.asarray(
        [lang_match_for_row(uid, lang) for uid, lang in zip(subs["user_id"].tolist(), subs["language"].tolist())],
        dtype=np.float32,
    )
    tag_match = np.asarray(
        [tag_match_for_row(uid, t2) for uid, t2 in zip(subs["user_id"].tolist(), tags2_list)],
        dtype=np.float32,
    )
    level = np.asarray([float(user_level.get(int(uid), 0.0)) for uid in subs["user_id"].tolist()], dtype=np.float32)
    perseverance = np.asarray(
        [float(user_pers.get(int(uid), 0.0)) for uid in subs["user_id"].tolist()], dtype=np.float32
    )

    lang_ohe = pd.get_dummies(subs["language"], prefix="lang")
    for l in lang_vocab:
        c = f"lang_{l}"
        if c not in lang_ohe.columns:
            lang_ohe[c] = 0
    lang_ohe = lang_ohe[[f"lang_{l}" for l in lang_vocab]]

    tag_to_j = {t: j for j, t in enumerate(tag_vocab)}
    tag_mh = np.zeros((len(subs), len(tag_vocab)), dtype=np.uint8)
    for i, t2 in enumerate(tags2_list):
        if not t2:
            continue
        for t in t2:
            j = tag_to_j.get(str(t))
            if j is not None:
                tag_mh[i, j] = 1
    tag_mh_df = pd.DataFrame(tag_mh, columns=[f"tag_{t}" for t in tag_vocab])

    out = pd.concat(
        [
            subs[["submission_id", "user_id", "problem_id", "attempt_no"]].astype(int),
            pd.DataFrame(
                {
                    "difficulty_filled": diff_filled.astype(int),
                    "level": level.astype(float),
                    "perseverance": perseverance.astype(float),
                    "lang_match": lang_match.astype(float),
                    "tag_match": tag_match.astype(float),
                }
            ),
            lang_ohe.astype(int),
            tag_mh_df.astype(int),
            subs[["ac"]].astype(int),
        ],
        axis=1,
    )
    return out


def build_models() -> dict[str, object]:
    """构建用于对比的基线模型集合（logreg/svm/tree）。"""
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, random_state=42)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "svm_linear": Pipeline(
            [("scaler", StandardScaler(with_mean=False)), ("clf", LinearSVC(random_state=42))]
        ),
    }


def eval_models(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """在给定时间切分下训练各模型并输出指标表。"""
    y = df["ac"].astype(int).to_numpy()
    X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"]).copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X_train, X_test = X[split.train_mask], X[split.test_mask]
    y_train, y_test = y[split.train_mask], y[split.test_mask]

    rows: list[dict] = []

    # baselines
    for strat in ("most_frequent", "stratified"):
        m = DummyClassifier(strategy=strat, random_state=42)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rows.append(
            {
                "model": f"dummy_{strat}",
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float("nan"),
                "brier": float("nan"),
            }
        )

    for name, model in build_models().items():
        model.fit(X_train.to_numpy(dtype=np.float32), y_train)
        pred = model.predict(X_test.to_numpy(dtype=np.float32))
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float("nan"),
            "brier": float("nan"),
        }

        # roc_auc & brier for logreg only
        if name == "logreg":
            prob = model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]
            row["roc_auc"] = float(roc_auc_score(y_test, prob))
            row["brier"] = float(brier_score_loss(y_test, prob))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model"])


def fit_logreg_probs(df: pd.DataFrame, split: Split) -> tuple[np.ndarray, np.ndarray]:
    """训练 logreg 并返回测试集 (y_true, p_ac) 用于画 ROC/PR/校准曲线。"""
    y = df["ac"].astype(int).to_numpy()
    X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"]).copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X_train, X_test = X[split.train_mask], X[split.test_mask]
    y_train, y_test = y[split.train_mask], y[split.test_mask]

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )
    model.fit(X_train.to_numpy(dtype=np.float32), y_train)
    prob = model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1].astype(np.float64)
    return y_test.astype(int), prob


def brier_decomposition(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10) -> dict[str, float]:
    """
    Brier Score 分解：reliability / resolution / uncertainty。

    说明
    - reliability：预测概率与真实频率的偏差（越小越好）
    - resolution：模型把样本分成不同风险层级的能力（越大越好）
    - uncertainty：标签本身的不确定性（由 base rate 决定）
    """
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    if len(p) == 0:
        return {
            "brier": float("nan"),
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
        }

    base = float(y.mean())
    uncertainty = base * (1.0 - base)
    brier = float(np.mean((p - y) ** 2))

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    reliability = 0.0
    resolution = 0.0
    for b in range(int(n_bins)):
        m = idx == b
        if not np.any(m):
            continue
        w = float(m.mean())
        p_bar = float(p[m].mean())
        o_bar = float(y[m].mean())
        reliability += w * (p_bar - o_bar) ** 2
        resolution += w * (o_bar - base) ** 2

    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def compare_feature_stats(strict_df: pd.DataFrame, leaky_df: pd.DataFrame) -> pd.DataFrame:
    """对比 strict/leaky 在核心特征上的分布统计（逐特征输出两行）。"""
    cols = ["difficulty_filled", "attempt_no", "level", "perseverance", "lang_match", "tag_match"]
    rows: list[dict] = []
    for c in cols:
        s = stat_series(strict_df[c])
        l = stat_series(leaky_df[c])
        rows.append({"feature": c, "variant": "strict", **s})
        rows.append({"feature": c, "variant": "leaky", **l})
    return pd.DataFrame(rows)


def per_user_level_variance(df: pd.DataFrame) -> pd.Series:
    """计算每个用户 level 的时间标准差（用于衡量画像是否“动态”）。"""
    g = df.groupby("user_id")["level"]
    return g.std(ddof=0)


def per_user_pers_variance(df: pd.DataFrame) -> pd.Series:
    """计算每个用户 perseverance 的时间标准差（用于衡量画像是否“动态”）。"""
    g = df.groupby("user_id")["perseverance"]
    return g.std(ddof=0)


def main() -> int:
    """CLI 入口：生成 strict vs leaky 的对比表格/图表/markdown 报告。"""
    parser = argparse.ArgumentParser(description="Compare strict (no-leak) vs leaky features and metrics.")
    parser.add_argument("--strict", default="FeatureData/train_samples.csv", help="Strict dataset CSV path")
    parser.add_argument("--submissions", default="CleanData/submissions.csv")
    parser.add_argument("--problems", default="CleanData/problems.csv")
    parser.add_argument("--students-derived", default="CleanData/students_derived.csv")
    parser.add_argument("--tags", default="CleanData/tags.csv")
    parser.add_argument("--languages", default="CleanData/languages.csv")
    # 表格/报告类产物输出到 Reports/compare；图表统一输出到 Reports/fig（便于 WebApp 展示）。
    parser.add_argument("--out-dir", default="Reports/compare")
    parser.add_argument("--time-split", type=float, default=0.8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (out_dir.parent / "fig").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    strict_df = pd.read_csv(args.strict, low_memory=False)
    strict_df["submission_id"] = pd.to_numeric(strict_df["submission_id"], errors="coerce").astype(int)
    strict_df["user_id"] = pd.to_numeric(strict_df["user_id"], errors="coerce").astype(int)
    strict_df["problem_id"] = pd.to_numeric(strict_df["problem_id"], errors="coerce").astype(int)
    strict_df["ac"] = pd.to_numeric(strict_df["ac"], errors="coerce").fillna(0).astype(int)

    split = time_split_by_submission_id(strict_df, frac=float(args.time_split))

    leaky_df = build_leaky_dataset(
        submissions_csv=args.submissions,
        problems_csv=args.problems,
        students_derived_csv=args.students_derived,
        tags_csv=args.tags,
        languages_csv=args.languages,
    )

    # align order by submission_id for fair split checks
    leaky_df["submission_id"] = pd.to_numeric(leaky_df["submission_id"], errors="coerce").astype(int)
    leaky_df = leaky_df.sort_values("submission_id").reset_index(drop=True)
    strict_df = strict_df.sort_values("submission_id").reset_index(drop=True)
    split = time_split_by_submission_id(strict_df, frac=float(args.time_split))

    # basic sanity: same rows / keys
    if len(leaky_df) != len(strict_df):
        raise SystemExit(f"row mismatch: strict={len(strict_df)} leaky={len(leaky_df)}")

    feat_stats = compare_feature_stats(strict_df, leaky_df)
    feat_stats.to_csv(out_dir / "compare_feature_stats.csv", index=False, encoding="utf-8-sig")

    lvl_std_strict = per_user_level_variance(strict_df)
    lvl_std_leaky = per_user_level_variance(leaky_df)
    pers_std_strict = per_user_pers_variance(strict_df)
    pers_std_leaky = per_user_pers_variance(leaky_df)

    # model metrics
    strict_metrics = eval_models(strict_df, split)
    strict_metrics.insert(0, "variant", "strict")
    leaky_metrics = eval_models(leaky_df, split)
    leaky_metrics.insert(0, "variant", "leaky")
    metrics = pd.concat([strict_metrics, leaky_metrics], axis=0, ignore_index=True)
    metrics.to_csv(out_dir / "compare_model_metrics.csv", index=False, encoding="utf-8-sig")

    def write_md_table(df: pd.DataFrame) -> str:
        view = df.copy()
        for c in view.columns:
            if pd.api.types.is_float_dtype(view[c]):
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        headers = [str(c) for c in view.columns]
        rows = [[str(x) for x in r] for r in view.to_numpy().tolist()]
        out_lines = []
        out_lines.append("| " + " | ".join(headers) + " |")
        out_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            out_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(out_lines) + "\n\n"

    def compute_reco_metrics(
            *,
            variant: str,
            df: pd.DataFrame,
            split: Split,
            submissions_csv: str,
            problems_csv: str,
            tags_csv: str,
            students_derived_csv: str,
            growth_min_p: float = 0.4,
            growth_max_p: float = 0.7,
            ks: tuple[int, ...] = (1, 3, 5, 10),
            chunk_size: int = 2048,
    ) -> pd.DataFrame:
        subs = pd.read_csv(submissions_csv, low_memory=False)
        subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
        subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
        subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
        subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)
        subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
        subs["language"] = subs.get("language", "").astype(str).fillna("")

        cutoff_id = int(split.cutoff_submission_id)
        subs_train = subs[subs["submission_id"] <= cutoff_id].copy()
        subs_test = subs[subs["submission_id"] > cutoff_id].copy()

        solved_before = subs_train[subs_train["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
        solved_map: dict[int, set[int]] = {
            int(uid): set(g["problem_id"].astype(int).tolist()) for uid, g in solved_before.groupby("user_id")
        }

        attempts_before = subs_train.groupby(["user_id", "problem_id"]).size().reset_index(name="n")
        attempt_next_map: dict[int, dict[int, int]] = {}
        for uid, g in attempts_before.groupby("user_id"):
            attempt_next_map[int(uid)] = {
                int(pid): int(n) + 1 for pid, n in zip(g["problem_id"], g["n"], strict=False)
            }

        test_ac = subs_test[subs_test["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
        test_ac_map: dict[int, set[int]] = {
            int(uid): set(g["problem_id"].astype(int).tolist()) for uid, g in test_ac.groupby("user_id")
        }
        users = sorted({int(x) for x in subs_train["user_id"].unique().tolist()})
        users_with_test = {int(x) for x in subs_test["user_id"].unique().tolist()}

        problems = pd.read_csv(problems_csv, low_memory=False)
        problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
        problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
        diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
        problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
        problems["tags_norm"] = problems["tags"].apply(parse_json_list)

        tag_whitelist = set(pd.read_csv(tags_csv)["tag_name"].astype(str).tolist())
        problems["tags_norm"] = problems["tags_norm"].apply(lambda lst: [t for t in lst if t in tag_whitelist])

        pids_arr = problems["problem_id"].to_numpy(dtype=np.int32)
        diffs_arr = problems["difficulty_filled"].to_numpy(dtype=np.int32)

        feature_cols = [c for c in df.columns if c not in {"ac", "submission_id", "user_id", "problem_id"}]
        col_to_idx = {c: i for i, c in enumerate(feature_cols)}
        required = {"attempt_no", "difficulty_filled", "level", "perseverance", "lang_match", "tag_match"}
        missing = sorted(required - set(feature_cols))
        if missing:
            raise RuntimeError(f"{variant}: missing required features: {missing}")

        lang_cols = [c for c in feature_cols if c.startswith("lang_")]
        tag_cols = [c for c in feature_cols if c.startswith("tag_")]
        lang_names = [c.removeprefix("lang_") for c in lang_cols]
        tag_names = [c.removeprefix("tag_") for c in tag_cols]
        tag_to_j = {t: j for j, t in enumerate(tag_names)}

        # train model on train window of df
        train_df = df[df["submission_id"].astype(int) <= cutoff_id].copy()
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["ac"].astype(int).to_numpy()
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, random_state=42)),
            ]
        )
        model.fit(X_train, y_train)

        # problem tag multi-hot for tag_match and tag_* features
        problem_tags_mh = np.zeros((len(problems), len(tag_names)), dtype=np.uint8)
        for i, tags_list in enumerate(problems["tags_norm"].tolist()):
            if not isinstance(tags_list, list):
                continue
            for t in tags_list[:2]:
                j = tag_to_j.get(str(t))
                if j is not None:
                    problem_tags_mh[i, j] = 1
        problem_tag_counts = problem_tags_mh.sum(axis=1).astype(np.float32)

        pid_to_i = {int(pid): i for i, pid in enumerate(pids_arr.tolist())}

        # user features (strict: subs_train; leaky: students_derived)
        if variant == "leaky":
            students = pd.read_csv(students_derived_csv, low_memory=False)
            students["user_id"] = pd.to_numeric(students["user_id"], errors="coerce").astype(int)
            students["level"] = pd.to_numeric(students["level"], errors="coerce").fillna(0.0).astype(float)
            students["perseverance"] = pd.to_numeric(students["perseverance"], errors="coerce").fillna(0.0).astype(
                float)
            students["lang_pref_dict"] = students["lang_pref"].apply(parse_json_dict)
            students["tag_pref_dict"] = students["tag_pref"].apply(parse_json_dict)
            user_level = students.set_index("user_id")["level"].to_dict()
            user_pers = students.set_index("user_id")["perseverance"].to_dict()
            user_lang_pref = students.set_index("user_id")["lang_pref_dict"].to_dict()
            user_tag_pref = students.set_index("user_id")["tag_pref_dict"].to_dict()
        else:
            up = subs_train.groupby(["user_id", "problem_id"], as_index=False).agg(
                n_attempts=("submission_id", "count"),
                solved=("ac", "max"),
            )
            up = up.merge(problems[["problem_id", "difficulty_filled"]], on="problem_id", how="left")
            up["difficulty_filled"] = pd.to_numeric(up["difficulty_filled"], errors="coerce").fillna(
                diff_median).astype(int)
            up["diff_norm"] = up["difficulty_filled"].astype(float) / 10.0
            num = (up["solved"].astype(float) * up["diff_norm"]).groupby(up["user_id"]).sum()
            den = up["diff_norm"].groupby(up["user_id"]).sum()
            user_level = (num / (den + 1e-9)).fillna(0.0).clip(0.0, 1.0).to_dict()

            avg_attempts_per_problem = (up.groupby("user_id")["n_attempts"].mean()).fillna(0.0).astype(float)
            p95 = float(np.percentile(avg_attempts_per_problem.values, 95)) if len(avg_attempts_per_problem) else 1.0
            denom_p = math.log1p(p95) if p95 > 0 else 1.0
            user_pers = (np.log1p(avg_attempts_per_problem) / (denom_p if denom_p > 0 else 1.0)).clip(0.0,
                                                                                                      1.0).to_dict()

            lang_counts = subs_train.groupby(["user_id", "language"]).size().reset_index(name="cnt")
            lang_tab = lang_counts.pivot_table(index="user_id", columns="language", values="cnt", fill_value=0)
            lang_tab = lang_tab.div(lang_tab.sum(axis=1).replace(0, 1), axis=0)
            user_lang_pref = {}
            for uid, row in lang_tab.iterrows():
                d = {}
                for l in lang_names:
                    v = float(row.get(l, 0.0))
                    if v > 0:
                        d[l] = v
                user_lang_pref[int(uid)] = d

            pid_to_tags_norm = {
                int(pid): (lst if isinstance(lst, list) else [])
                for pid, lst in zip(problems["problem_id"].astype(int), problems["tags_norm"].tolist())
            }
            tag_rows: list[tuple[int, str]] = []
            for uid, pid in up[["user_id", "problem_id"]].itertuples(index=False):
                for t in pid_to_tags_norm.get(int(pid), [])[:2]:
                    tag_rows.append((int(uid), str(t)))
            if tag_rows:
                tag_df = pd.DataFrame(tag_rows, columns=["user_id", "tag"])
                tag_counts = tag_df.groupby(["user_id", "tag"]).size().reset_index(name="cnt")
                tag_tab = tag_counts.pivot_table(index="user_id", columns="tag", values="cnt", fill_value=0)
                tag_tab = tag_tab.div(tag_tab.sum(axis=1).replace(0, 1), axis=0)
                user_tag_pref = {
                    int(uid): {t: float(row.get(t, 0.0)) for t in tag_names if float(row.get(t, 0.0)) > 0}
                    for uid, row in tag_tab.iterrows()
                }
            else:
                user_tag_pref = {}

        def top_language(uid: int) -> tuple[str, float]:
            pref = user_lang_pref.get(uid, {}) or {}
            if not lang_names:
                return "", 0.0
            if not pref:
                return lang_names[0], 0.0
            best_l = lang_names[0]
            best_p = -1.0
            for l in lang_names:
                p = float(pref.get(l, 0.0))
                if p > best_p:
                    best_p = p
                    best_l = l
            return best_l, max(0.0, float(best_p))

        max_k = max(ks)
        rows: list[dict] = []

        for uid in users:
            solved = solved_map.get(uid, set())
            if len(solved) >= len(pids_arr):
                continue

            l, lprob = top_language(uid)
            level = float(user_level.get(uid, 0.0))
            pers = float(user_pers.get(uid, 0.0))
            tpref = user_tag_pref.get(uid, {}) or {}
            tag_pref_vec = np.asarray([float(tpref.get(t, 0.0)) for t in tag_names], dtype=np.float32)
            tm_sum = (problem_tags_mh.astype(np.float32) * tag_pref_vec).sum(axis=1)
            tm_den = np.maximum(1.0, problem_tag_counts)
            tag_match_vec = (tm_sum / tm_den).astype(np.float32)

            attempt_no_vec = np.ones((len(pids_arr),), dtype=np.float32)
            for pid, nxt in (attempt_next_map.get(uid, {}) or {}).items():
                i = pid_to_i.get(int(pid))
                if i is not None:
                    attempt_no_vec[i] = float(max(1, min(10, int(nxt))))

            lang_one_hot = np.zeros((len(lang_cols),), dtype=np.float32)
            if l in lang_names and lang_cols:
                lang_one_hot[lang_names.index(l)] = 1.0

            probs = np.zeros((len(pids_arr),), dtype=np.float32)
            for start in range(0, len(pids_arr), chunk_size):
                end = min(len(pids_arr), start + chunk_size)
                X = np.zeros((end - start, len(feature_cols)), dtype=np.float32)
                X[:, col_to_idx["attempt_no"]] = attempt_no_vec[start:end]
                X[:, col_to_idx["difficulty_filled"]] = diffs_arr[start:end].astype(np.float32)
                X[:, col_to_idx["level"]] = float(level)
                X[:, col_to_idx["perseverance"]] = float(pers)
                X[:, col_to_idx["lang_match"]] = float(lprob)
                X[:, col_to_idx["tag_match"]] = tag_match_vec[start:end]
                if tag_cols:
                    X[:, [col_to_idx[c] for c in tag_cols]] = problem_tags_mh[start:end].astype(np.float32)
                if lang_cols:
                    X[:, [col_to_idx[c] for c in lang_cols]] = lang_one_hot
                probs[start:end] = model.predict_proba(X)[:, 1].astype(np.float32)

            in_band = (probs >= growth_min_p) & (probs <= growth_max_p)
            candidate_mask = ~np.isin(pids_arr,
                                      np.fromiter((int(x) for x in solved), dtype=np.int32)) if solved else np.ones(
                (len(pids_arr),), dtype=bool)
            candidate_idx = np.where(candidate_mask)[0]
            band_idx = candidate_idx[in_band[candidate_idx]]
            other_idx = candidate_idx[~in_band[candidate_idx]]
            band_sorted = band_idx[np.argsort(probs[band_idx])[::-1]]
            other_sorted = other_idx[np.argsort(probs[other_idx])[::-1]]
            chosen = np.concatenate([band_sorted, other_sorted], axis=0)[:max_k]
            rec_pids = [int(pids_arr[i]) for i in chosen.tolist()]

            gt = test_ac_map.get(uid, set())
            if not gt:
                continue

            for k in ks:
                topk = set(rec_pids[:k])
                inter = len(topk & gt)
                rows.append(
                    {
                        "variant": variant,
                        "user_id": uid,
                        "k": int(k),
                        "hit": int(inter > 0),
                        "precision": float(inter / float(k)),
                        "active": int(uid in users_with_test),
                    }
                )

        metrics_rows: list[dict] = []
        mdf = pd.DataFrame(rows)
        for k in ks:
            sub = mdf[mdf["k"] == int(k)]
            sub_active = sub[sub["active"] == 1]
            metrics_rows.append(
                {
                    "variant": variant,
                    "k": int(k),
                    "hit_at_k_all": float(sub["hit"].mean()) if len(sub) else 0.0,
                    "precision_at_k_all": float(sub["precision"].mean()) if len(sub) else 0.0,
                    "hit_at_k_active": float(sub_active["hit"].mean()) if len(sub_active) else 0.0,
                    "precision_at_k_active": float(sub_active["precision"].mean()) if len(sub_active) else 0.0,
                    "users_all": int(sub["user_id"].nunique()) if len(sub) else 0,
                    "users_active": int(sub_active["user_id"].nunique()) if len(sub_active) else 0,
                    "growth_band": f"[{growth_min_p},{growth_max_p}]",
                    "cutoff_submission_id": cutoff_id,
                }
            )
        return pd.DataFrame(metrics_rows)

    # write report
    report_path = out_dir / "compare_strict_vs_leaky_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Strict（无泄漏） vs Leaky（泄漏）对比报告\n\n")
        f.write(f"- strict: `{args.strict}`\n")
        f.write(f"- leaky: 由 `{args.submissions}` + `{args.students_derived}` 重建（等价于旧版特征口径）\n")
        f.write(f"- time split: {args.time_split:.2f}（cutoff_submission_id={split.cutoff_submission_id}）\n\n")

        def w_table(df: pd.DataFrame) -> None:
            f.write(write_md_table(df))

        f.write("## 1) 关键特征分布（均值/中位数/P95/标准差）\n\n")
        w_table(feat_stats)

        f.write("## 2) 同一用户画像随时间变化的量化（std 越大表示越动态）\n\n")
        summary = pd.DataFrame(
            [
                {
                    "feature": "level",
                    "variant": "strict",
                    "mean_user_std": float(lvl_std_strict.mean()),
                    "p50_user_std": float(lvl_std_strict.median()),
                    "p95_user_std": float(np.percentile(lvl_std_strict.to_numpy(dtype=float), 95)),
                },
                {
                    "feature": "level",
                    "variant": "leaky",
                    "mean_user_std": float(lvl_std_leaky.mean()),
                    "p50_user_std": float(lvl_std_leaky.median()),
                    "p95_user_std": float(np.percentile(lvl_std_leaky.to_numpy(dtype=float), 95)),
                },
                {
                    "feature": "perseverance",
                    "variant": "strict",
                    "mean_user_std": float(pers_std_strict.mean()),
                    "p50_user_std": float(pers_std_strict.median()),
                    "p95_user_std": float(np.percentile(pers_std_strict.to_numpy(dtype=float), 95)),
                },
                {
                    "feature": "perseverance",
                    "variant": "leaky",
                    "mean_user_std": float(pers_std_leaky.mean()),
                    "p50_user_std": float(pers_std_leaky.median()),
                    "p95_user_std": float(np.percentile(pers_std_leaky.to_numpy(dtype=float), 95)),
                },
            ]
        )
        w_table(summary)

        f.write("## 3) 模型效果对比（同一切分规则）\n\n")
        w_table(metrics.sort_values(["model", "variant"]))

        f.write("## 4) 推荐评估对比（Hit@K / Precision@K）\n\n")
        reco_leaky = compute_reco_metrics(
            variant="leaky",
            df=leaky_df,
            split=split,
            submissions_csv=args.submissions,
            problems_csv=args.problems,
            tags_csv=args.tags,
            students_derived_csv=args.students_derived,
        )
        reco_strict = compute_reco_metrics(
            variant="strict",
            df=strict_df,
            split=split,
            submissions_csv=args.submissions,
            problems_csv=args.problems,
            tags_csv=args.tags,
            students_derived_csv=args.students_derived,
        )
        reco = pd.concat([reco_strict, reco_leaky], axis=0, ignore_index=True)
        reco.to_csv(out_dir / "compare_reco_metrics.csv", index=False, encoding="utf-8-sig")
        w_table(reco.sort_values(["k", "variant"]))

        # plots
        try:
            pivot_hit = reco.pivot(index="k", columns="variant", values="hit_at_k_all")
            pivot_prec = reco.pivot(index="k", columns="variant", values="precision_at_k_all")
            plt.figure(figsize=(6.5, 4.0))
            for col in pivot_hit.columns:
                plt.plot(pivot_hit.index, pivot_hit[col], marker="o", label=f"{col}")
            plt.title("Hit@K (all users)")
            plt.xlabel("K")
            plt.ylabel("Hit@K")
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_compare_hitk.png", dpi=200)
            plt.close()

            plt.figure(figsize=(6.5, 4.0))
            for col in pivot_prec.columns:
                plt.plot(pivot_prec.index, pivot_prec[col], marker="o", label=f"{col}")
            plt.title("Precision@K (all users)")
            plt.xlabel("K")
            plt.ylabel("Precision@K")
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_compare_precisionk.png", dpi=200)
            plt.close()
        except Exception:
            pass

        f.write("## 5) 概率质量诊断（ROC/PR/校准/Brier 分解）\n\n")

        y_strict, p_strict = fit_logreg_probs(strict_df, split)
        y_leaky, p_leaky = fit_logreg_probs(leaky_df, split)

        # ROC curve
        fpr_s, tpr_s, _ = roc_curve(y_strict, p_strict)
        fpr_l, tpr_l, _ = roc_curve(y_leaky, p_leaky)
        roc_auc_s = float(auc(fpr_s, tpr_s))
        roc_auc_l = float(auc(fpr_l, tpr_l))
        pd.DataFrame(
            {
                "fpr_strict": pd.Series(fpr_s),
                "tpr_strict": pd.Series(tpr_s),
                "fpr_leaky": pd.Series(fpr_l),
                "tpr_leaky": pd.Series(tpr_l),
            }
        ).to_csv(out_dir / "compare_roc_curve.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(6.5, 4.4))
        plt.plot(fpr_s, tpr_s, lw=2, label=f"strict (AUC={roc_auc_s:.4f})")
        plt.plot(fpr_l, tpr_l, lw=2, label=f"leaky (AUC={roc_auc_l:.4f})")
        plt.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
        plt.title("ROC curve (logreg)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_compare_roc.png", dpi=200)
        plt.close()

        # Precision-Recall curve
        prec_s, rec_s, _ = precision_recall_curve(y_strict, p_strict)
        prec_l, rec_l, _ = precision_recall_curve(y_leaky, p_leaky)
        ap_s = float(average_precision_score(y_strict, p_strict))
        ap_l = float(average_precision_score(y_leaky, p_leaky))
        pd.DataFrame(
            {
                "precision_strict": pd.Series(prec_s),
                "recall_strict": pd.Series(rec_s),
                "precision_leaky": pd.Series(prec_l),
                "recall_leaky": pd.Series(rec_l),
            }
        ).to_csv(out_dir / "compare_pr_curve.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(6.5, 4.4))
        plt.plot(rec_s, prec_s, lw=2, label=f"strict (AP={ap_s:.4f})")
        plt.plot(rec_l, prec_l, lw=2, label=f"leaky (AP={ap_l:.4f})")
        plt.title("Precision-Recall curve (logreg)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_compare_pr.png", dpi=200)
        plt.close()

        # Calibration curve (reliability diagram)
        frac_s, mean_s = calibration_curve(y_strict, p_strict, n_bins=10, strategy="uniform")
        frac_l, mean_l = calibration_curve(y_leaky, p_leaky, n_bins=10, strategy="uniform")
        pd.DataFrame(
            {
                "mean_pred_strict": pd.Series(mean_s),
                "frac_pos_strict": pd.Series(frac_s),
                "mean_pred_leaky": pd.Series(mean_l),
                "frac_pos_leaky": pd.Series(frac_l),
            }
        ).to_csv(out_dir / "compare_calibration_curve.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(6.5, 4.4))
        plt.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1, label="perfect")
        plt.plot(mean_s, frac_s, marker="o", lw=2, label="strict")
        plt.plot(mean_l, frac_l, marker="o", lw=2, label="leaky")
        plt.title("Calibration curve (logreg)")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_compare_calibration.png", dpi=200)
        plt.close()

        # Brier decomposition
        bd_s = brier_decomposition(y_strict, p_strict, n_bins=10)
        bd_l = brier_decomposition(y_leaky, p_leaky, n_bins=10)
        cal_metrics = pd.DataFrame(
            [
                {"variant": "strict", "roc_auc": roc_auc_s, "ap": ap_s, **bd_s},
                {"variant": "leaky", "roc_auc": roc_auc_l, "ap": ap_l, **bd_l},
            ]
        )
        cal_metrics.to_csv(out_dir / "compare_calibration_metrics.csv", index=False, encoding="utf-8-sig")
        w_table(cal_metrics)

        f.write("## 结论建议（如何判断“是否失真”）\n\n")
        f.write("- 如果 leaky 显著高于 strict，说明旧口径存在“看未来”的时间泄漏，评估被抬高。\n")
        f.write("- strict 指标更接近线上真实可用效果；推荐评估也应优先看 strict 版本。\n")
        f.write("\n")

    print("Wrote", report_path)
    print("Wrote", out_dir / "compare_feature_stats.csv")
    print("Wrote", out_dir / "compare_model_metrics.csv")
    print("Wrote", out_dir / "compare_reco_metrics.csv")
    print("Wrote", out_dir / "compare_calibration_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
