"""
recommendation.py

训练脚本：完成学生画像派生、训练样本构建、模型训练与模型持久化。

输入：
- CleanData/problems.csv
- CleanData/submissions.csv（或 submissions_clean.csv）
- CleanData/tags.csv
- CleanData/languages.csv

输出：
- CleanData/students_derived.csv
- FeatureData/train_samples.csv
- Models/pipeline_logreg.joblib
- Models/reco_logreg.joblib
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

# 以当前文件目录作为根目录，统一管理相对路径
ROOT = Path(__file__).resolve().parent

# 输入路径（CleanData）
PROBLEMS = ROOT / "CleanData/problems.csv"
SUBMISSIONS = ROOT / "CleanData/submissions.csv"
SUBMISSIONS_COMPAT = ROOT / "CleanData/submissions_clean.csv"
TAGS = ROOT / "CleanData/tags.csv"
LANGS = ROOT / "CleanData/languages.csv"

# 输出路径（派生特征、训练样本与模型）
STUDENTS_OUT = ROOT / "CleanData/students_derived.csv"
TRAIN_SAMPLES_OUT = ROOT / "FeatureData/train_samples.csv"
PIPELINE_OUT = ROOT / "Models/pipeline_logreg.joblib"
RECO_MODEL_OUT = ROOT / "Models/reco_logreg.joblib"

# 训练随机种子与时间切分比例
RANDOM_SEED = 42
TIME_SPLIT = 0.8


def parse_json_list(x: object) -> list[str]:
    """
    将 CSV 单元格解析为字符串列表。

    支持以下输入：
    - Python list
    - JSON 列表字符串
    - 逗号/分号/竖线分隔字符串
    - None / NaN / 空字符串
    """
    # 空值直接返回空列表
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []

    # 若已是 list，直接转字符串
    if isinstance(x, list):
        return [str(t) for t in x]

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []

    # 优先尝试 JSON 列表解析
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass

    # 回退为分隔符切分
    s = s.strip("[]")
    parts = re.split(r"[;,]\s*|\s+\|\s+|\s+,\s+", s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def first_existing_path(*candidates: Path) -> Path:
    """返回候选路径中第一个存在的文件路径。"""
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
    """
    由题目、提交、标签与语言数据派生学生画像，并保存到 CSV。

    输出字段包含：
    - level：难度加权的完成度
    - perseverance：平均尝试次数的对数归一化
    - lang_pref：语言偏好分布（JSON）
    - tag_pref：标签偏好分布（JSON）
    """
    # 读取输入数据
    problems = pd.read_csv(problems_path)
    subs = pd.read_csv(submissions_path)
    tags = pd.read_csv(tags_path)
    langs = pd.read_csv(langs_path)

    # 题目侧：难度填充 + 标签规范化
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    # 仅保留标签词表中的标签，且每题最多 2 个标签
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    problems["tags_norm"] = problems["tags_list"].apply(
        lambda lst: [t for t in lst if t in tag_set][:2]
    )

    # 提交侧：AC 标记转为 0/1
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    # user-problem 聚合：同一用户同题的多次提交合并
    up = (
        subs.groupby(["user_id", "problem_id"], as_index=False)
        .agg(n_attempts=("submission_id", "count"), solved=("ac", "max"))
        .merge(problems[["problem_id", "difficulty_filled", "tags_norm"]], on="problem_id", how="left")
    )

    # 缺失难度补全，并归一化到 0~1
    up["difficulty_filled"] = up["difficulty_filled"].fillna(diff_median).astype(int)
    up["diff_norm"] = up["difficulty_filled"] / 10.0

    # level：历史已 AC 的难度强度 / 历史尝试过的难度强度
    num = (up["solved"] * up["diff_norm"]).groupby(up["user_id"]).sum()
    den = up["diff_norm"].groupby(up["user_id"]).sum()
    level = (num / (den + 1e-9)).reset_index(name="level")

    # perseverance：平均每题尝试次数，log1p 后按全局 P95 归一化
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

    # lang_pref：语言使用比例分布（仅统计词表内语言）
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

    # tag_pref：基于 user-problem 去重后的标签偏好分布
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

    # 汇总：确保出现过提交的用户都有画像
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

    # 保存结果
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
    """
    以每条 submission 构造训练样本，严格使用“历史信息”生成特征。

    输出字段包含：
    - 基础键：submission_id / user_id / problem_id / attempt_no
    - 数值特征：difficulty_filled / level / perseverance / lang_match / tag_match
    - 稀疏特征：语言 one-hot、标签 multi-hot
    - 标签：ac
    """
    # 读取输入数据
    problems = pd.read_csv(problems_path)
    subs = pd.read_csv(submissions_path, low_memory=False)
    tags = pd.read_csv(tags_path)
    langs = pd.read_csv(langs_path)

    # 词表：固定列顺序，确保训练可复现
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    lang_vocab = sorted(set(langs["name"].astype(str).tolist()))
    lang_set = set(lang_vocab)

    # 题目表预处理：难度填充 + 标签解析
    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["diff_norm"] = problems["difficulty_filled"].astype(float) / 10.0
    problems["tags_list"] = problems["tags"].apply(parse_json_list)
    problems["tags2"] = problems["tags_list"].apply(lambda lst: [t for t in lst if t in tag_set][:2])

    # 题目静态信息映射表：加速后续按 problem_id 查询
    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int)))
    pid_to_diffnorm = dict(zip(problems["problem_id"].astype(int), problems["diff_norm"].astype(float)))
    pid_to_tags2 = dict(zip(problems["problem_id"].astype(int), problems["tags2"].tolist()))

    # 提交表字段清洗
    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)
    subs["language"] = subs.get("language", "").astype(str).fillna("")

    # 按 user_id + submission_id 排序，确保“只看过去”
    subs = subs.sort_values(["user_id", "submission_id"]).reset_index(drop=True)

    # 计算全局归一化基准（P95）
    user_total = subs.groupby("user_id").size()
    user_unique_prob = subs.groupby("user_id")["problem_id"].nunique()
    avg_attempts = (user_total / user_unique_prob.replace(0, np.nan)).dropna().values.astype(float)
    p95 = float(np.percentile(avg_attempts, 95)) if len(avg_attempts) else 1.0
    denom_p = math.log1p(p95) if p95 > 0 else 1.0

    # 预分配数组，提升特征构造效率
    n = len(subs)
    level_arr = np.zeros((n,), dtype=np.float32)
    pers_arr = np.zeros((n,), dtype=np.float32)
    lang_match_arr = np.zeros((n,), dtype=np.float32)
    tag_match_arr = np.zeros((n,), dtype=np.float32)

    # 按 submission 顺序对齐题目静态属性
    tags2_list = subs["problem_id"].map(lambda pid: pid_to_tags2.get(int(pid), [])).to_list()
    diff_filled = subs["problem_id"].map(lambda pid: pid_to_diff.get(int(pid), diff_median)).astype(int)
    diff_norm = subs["problem_id"].map(lambda pid: pid_to_diffnorm.get(int(pid), diff_median / 10.0)).astype(float)

    tag_to_j = {t: j for j, t in enumerate(tag_vocab)}

    # 逐用户滚动扫描：每条样本只使用该用户历史状态
    for uid, idx in subs.groupby("user_id", sort=False).groups.items():
        idx_list = idx.tolist()

        # 用户历史状态（会随着时间推进不断更新）
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

            # ---- 使用“历史状态”生成当前样本特征 ----
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

            # ---- 将“当前提交”计入历史状态，供后续样本使用 ----
            total_subs += 1
            if lang in lang_set:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            # 同题首次尝试时，才计入难度与标签的分母
            if pid not in problems_attempted:
                problems_attempted.add(pid)
                denom_sum += float(diff_norm.iat[i])
                for t in t2:
                    if t in tag_to_j:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                        total_tag += 1

            # 同题首次 AC 时，才计入难度分子
            if ac == 1 and pid not in problems_solved:
                problems_solved.add(pid)
                num_sum += float(diff_norm.iat[i])

    # 语言 one-hot：保持列全集，确保训练列一致
    lang_ohe = pd.get_dummies(subs["language"], prefix="lang")
    for l in lang_vocab:
        c = f"lang_{l}"
        if c not in lang_ohe.columns:
            lang_ohe[c] = 0
    lang_ohe = lang_ohe[[f"lang_{l}" for l in lang_vocab]]

    # 标签 multi-hot：每题最多 2 个标签
    tag_mh = np.zeros((n, len(tag_vocab)), dtype=np.uint8)
    for i, t2 in enumerate(tags2_list):
        if not t2:
            continue
        for t in t2:
            j = tag_to_j.get(t)
            if j is not None:
                tag_mh[i, j] = 1
    tag_mh_df = pd.DataFrame(tag_mh, columns=[f"tag_{t}" for t in tag_vocab])

    # 汇总输出：基础键 + 特征 + 标签
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

    # 保存训练样本
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Wrote", str(out_path), "rows=", len(out), "cols=", out.shape[1])


def load_train_samples(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    读取训练样本并拆分出特征、标签与时间排序键。

    返回：
    - X：特征表
    - y：标签数组（0/1）
    - submission_id：用于时间切分
    - feature_cols：特征列名列表
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Missing train samples: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"ac", "submission_id", "user_id", "problem_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Train samples missing columns: {sorted(missing)}")

    # 标签与时间键
    y = df["ac"].astype(int).to_numpy()
    submission_id = df["submission_id"].to_numpy()

    # 特征列：去除标签与 ID 字段
    X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"]).copy()
    feature_cols = list(X.columns)

    # 将 bool 转为 0/1，保持数值一致性
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
    """
    按 submission_id 进行时间切分，前 train_ratio 为训练集。
    """
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
    """构建候选模型集合（仅训练，不做评估输出）。"""
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
    """训练候选模型，用于确认基础训练流程可完成。"""
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
    """
    使用全量样本训练逻辑回归，并保存可直接推理使用的 Pipeline。
    """
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
    """
    训练推荐阶段使用的逻辑回归模型，并保存模型与元信息。

    该模型仅使用时间切分前的训练窗口数据。
    """
    df = pd.read_csv(train_samples_path)
    df["submission_id"] = pd.to_numeric(df["submission_id"], errors="coerce").astype(int)
    df = df.sort_values("submission_id").reset_index(drop=True)

    # 特征列 = 去除标签与 ID 字段后的所有列
    feature_cols = [
        c for c in df.columns if c not in {"ac", "submission_id", "user_id", "problem_id"}
    ]
    if not feature_cols:
        raise ValueError("No feature columns found for recommendation model.")

    # 依据时间切分比例确定训练窗口
    split = int(len(df) * train_ratio)
    if split <= 0:
        raise ValueError("Not enough rows for recommendation training split.")
    cutoff_id = int(df["submission_id"].iloc[split - 1])

    train_df = df.iloc[:split].reset_index(drop=True)
    X_train = train_df[feature_cols].copy()
    y_train = train_df["ac"].astype(int).to_numpy()

    # bool 列统一转为 0/1
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
    """训练主流程：派生画像 -> 构建样本 -> 训练模型 -> 保存产物。"""
    np.random.seed(RANDOM_SEED)

    # 兼容两种提交文件名：优先使用存在的文件
    submissions_path = first_existing_path(SUBMISSIONS, SUBMISSIONS_COMPAT)

    # Step 1：派生学生画像
    print("Step 1/4: derive student profiles")
    derive_students(
        problems_path=PROBLEMS,
        submissions_path=submissions_path,
        tags_path=TAGS,
        langs_path=LANGS,
        out_path=STUDENTS_OUT,
    )

    # Step 2：构建训练样本
    print("Step 2/4: build training samples")
    build_train_samples(
        problems_path=PROBLEMS,
        submissions_path=submissions_path,
        tags_path=TAGS,
        langs_path=LANGS,
        out_path=TRAIN_SAMPLES_OUT,
    )

    # Step 3：训练基线模型与最终推理 Pipeline
    print("Step 3/4: train baseline models and web pipeline")
    X, y, submission_id, feature_cols = load_train_samples(TRAIN_SAMPLES_OUT)
    X_train, _X_test, y_train, _y_test = time_split_by_submission_id(
        X, y, submission_id, train_ratio=0.8
    )
    train_baseline_models(X_train, y_train)
    save_webapp_pipeline(X=X, y=y, feature_cols=feature_cols, out_path=PIPELINE_OUT, random_seed=RANDOM_SEED)

    # Step 4：训练推荐模型
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
