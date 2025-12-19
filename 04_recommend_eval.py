"""
04_recommend_eval.py

离线推荐评估脚本：
- 先用 `FeatureData/train_samples.csv` 训练一个简单的 AC 概率模型（逻辑回归）。
- 再以 `CleanData/submissions.csv` 的时间切分窗口构造用户画像与 ground truth，
  生成 Top‑K 推荐并计算 Hit@K / Precision@K / Recall@K / NDCG@K 等指标。

关键约束：严格无泄漏
- 用户画像/偏好统计只使用训练窗口（cutoff 之前）的提交。
- ground truth 只使用测试窗口（cutoff 之后）的 AC。

输出到 `Reports/reco/`：
- recommendations_topk.csv：仅保留 `model_growth`（兼容旧输出）
- recommendations_topk_compare.csv：所有策略明细
- reco_metrics.csv：指标汇总
输出到 `Reports/fig/`：
- fig_命中率曲线.png / fig_推荐集中度与覆盖率.png / fig_推荐难度分布_单用户.png：图表
"""

import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 将 matplotlib 缓存目录落到仓库内，避免在无 HOME/只读环境下写缓存失败
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CleanData / FeatureData 输入
SUBMISSIONS = ROOT / "CleanData/submissions.csv"
PROBLEMS = ROOT / "CleanData/problems.csv"
TAGS = ROOT / "CleanData/tags.csv"
TRAIN_SAMPLES = ROOT / "FeatureData/train_samples.csv"

# Reports 输出
OUT_RECO = ROOT / "Reports/reco/recommendations_topk.csv"
OUT_RECO_COMPARE = ROOT / "Reports/reco/recommendations_topk_compare.csv"
OUT_METRICS = ROOT / "Reports/reco/reco_metrics.csv"
FIG_HITK = ROOT / "Reports/fig/fig_命中率曲线.png"
FIG_CASE_DIFF = ROOT / "Reports/fig/fig_推荐难度分布_单用户.png"
FIG_COVERAGE = ROOT / "Reports/fig/fig_推荐集中度与覆盖率.png"

# 时间切分：按 submission_id 排序后前 80% 作为训练窗口
TIME_SPLIT = 0.8

# Top‑K 评估的 K 列表
KS = (1, 3, 5, 10)
MAX_K = max(KS)

# “成长带”概率区间：用于 growth 策略
GROWTH_MIN_P = 0.4
GROWTH_MAX_P = 0.7

# 打分时对候选题分块，避免一次性构造过大的矩阵导致内存峰值过高
CHUNK_SIZE = 2048
RANDOM_SEED = 42

# 参与对比的策略名（同时用于输出与指标汇总）
STRATEGIES = (
    "model_maxprob",
    "model_growth",
    "popular_train",
    "easy_first",
    "random",
)


def setup_cn_font() -> None:
    """尽可能配置中文字体，保证图表标题/坐标轴中文可正常显示。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


setup_cn_font()


def parse_json_dict(x) -> dict[str, float]:
    """把 JSON 字符串/字典解析为 {str: float}；解析失败返回空 dict。"""
    if x is None:
        return {}
    if isinstance(x, dict):
        return {str(k): float(v) for k, v in x.items()}
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        v = json.loads(s)
    except Exception:
        return {}
    if not isinstance(v, dict):
        return {}
    out: dict[str, float] = {}
    for k, val in v.items():
        try:
            out[str(k)] = float(val)
        except Exception:
            continue
    return out


def parse_json_list(x) -> list[str]:
    """
    将 CSV 单元格解析为字符串列表（用于题目 tags 字段）。

    支持 JSON 列表字符串或常见分隔符（逗号/分号/竖线）形式。
    """
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
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


def ensure_dir(path: Path) -> None:
    """确保输出文件所在目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def ndcg_at_k(rec_pids: list[int], gt: set[int], k: int) -> float:
    """
    计算 NDCG@K（相关性为二值：是否在 ground truth 中）。

    - rec_pids：推荐列表（按 rank 从高到低）
    - gt：ground truth 集合（测试窗口内用户 AC 过的题）
    """
    k = int(k)
    if k <= 0:
        return 0.0
    if not gt:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(rec_pids[:k]):
        if pid in gt:
            dcg += 1.0 / math.log2(i + 2.0)
    ideal = 0.0
    for i in range(min(k, len(gt))):
        ideal += 1.0 / math.log2(i + 2.0)
    return float(dcg / ideal) if ideal > 0 else 0.0


def main() -> int:
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # 准备输出目录
    ensure_dir(OUT_RECO)
    ensure_dir(OUT_RECO_COMPARE)
    ensure_dir(OUT_METRICS)
    ensure_dir(FIG_HITK)
    ensure_dir(FIG_CASE_DIFF)
    ensure_dir(FIG_COVERAGE)

    # 读取训练样本（来自 02_build_features.py）
    df = pd.read_csv(TRAIN_SAMPLES)
    df["submission_id"] = pd.to_numeric(df["submission_id"], errors="coerce").astype(int)

    # 特征列 = train_samples 中除去键与 label 的所有列（列顺序会影响模型输入）
    feature_cols = [
        c for c in df.columns if c not in {"ac", "submission_id", "user_id", "problem_id"}
    ]
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}

    # 这些列用于后续“线上打分”时构造特征向量，必须存在
    required = {"attempt_no", "difficulty_filled", "level", "perseverance", "lang_match", "tag_match"}
    missing = sorted(required - set(feature_cols))
    if missing:
        raise SystemExit(f"train_samples 缺少必要特征列：{missing}")

    # 稀疏列：语言 one‑hot / 标签 multi‑hot
    lang_cols = [c for c in feature_cols if c.startswith("lang_")]
    tag_cols = [c for c in feature_cols if c.startswith("tag_")]
    lang_names = [c.removeprefix("lang_") for c in lang_cols]
    tag_names = [c.removeprefix("tag_") for c in tag_cols]

    # 按 submission_id 排序做时间切分（同一 cutoff 也用于 CleanData/submissions.csv 的 train/test 切窗）
    order = np.argsort(df["submission_id"].values)
    df = df.iloc[order].reset_index(drop=True)
    split = int(len(df) * TIME_SPLIT)
    cutoff_id = int(df["submission_id"].iloc[split - 1])

    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["ac"].astype(int).values
    # 注意：这里不直接用 test_df 做分类指标；推荐评估使用 CleanData/submissions.csv 的 test 窗口作为 ground truth

    # 一个轻量模型：标准化 + 逻辑回归（with_mean=False 以兼容 one-hot 特征）
    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(X_train.to_numpy(dtype=np.float32), y_train)

    # 读取提交记录，用同一 cutoff_id 划分 train/test 窗口（用于构建画像与评估）
    subs = pd.read_csv(SUBMISSIONS)
    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    subs_train = subs[subs["submission_id"] <= cutoff_id].copy()
    subs_test = subs[subs["submission_id"] > cutoff_id].copy()

    # active users：在测试窗口中至少有一次提交的用户（更接近线上实际）
    users_with_test = {int(x) for x in subs_test["user_id"].unique().tolist()}

    # cutoff 之前已 AC 的题：推荐时剔除，避免“推荐已解题”
    solved_before = (
        subs_train[subs_train["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
    )
    solved_map: dict[int, set[int]] = {}
    for uid, g in solved_before.groupby("user_id"):
        solved_map[int(uid)] = set(g["problem_id"].astype(int).tolist())

    # 训练窗内每个 user-problem 的提交次数，用于推断“下一次 attempt_no”（没做过则为 1）
    attempts_before = subs_train.groupby(["user_id", "problem_id"]).size().reset_index(name="n")
    attempt_next_map: dict[int, dict[int, int]] = {}
    for uid, g in attempts_before.groupby("user_id"):
        attempt_next_map[int(uid)] = {
            # zip(strict=False) 需要 Python 3.10+；这里是 1:1 对齐，不额外做严格校验
            int(pid): int(n) + 1 for pid, n in zip(g["problem_id"], g["n"], strict=False)
        }

    # ground truth：测试窗内 AC 过的题（去重），用于 Hit@K / Recall@K / NDCG@K
    test_ac = subs_test[subs_test["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
    test_ac_map: dict[int, set[int]] = {}
    for uid, g in test_ac.groupby("user_id"):
        test_ac_map[int(uid)] = set(g["problem_id"].astype(int).tolist())
    users_with_test_ac = set(test_ac_map.keys())

    # 题目表：用于候选集合、难度与 tags multi-hot（维度需与 train_samples 的 tag_* 列一致）
    problems = pd.read_csv(PROBLEMS)
    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_norm"] = problems["tags"].apply(parse_json_list)

    # tags 白名单：与 CleanData/tags.csv 对齐，过滤掉脏标签/未知标签
    tag_whitelist = set(pd.read_csv(TAGS)["tag_name"].astype(str).tolist())
    problems["tags_norm"] = problems["tags_norm"].apply(lambda lst: [t for t in lst if t in tag_whitelist])

    # numpy 化以加速后续批量打分
    problem_ids = problems["problem_id"].to_numpy(dtype=np.int32)
    problem_diff = problems["difficulty_filled"].to_numpy(dtype=np.int32)
    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int), strict=False))

    # 构造题目 tags multi-hot：列顺序必须与训练特征 tag_* 一致
    tag_to_j = {t: j for j, t in enumerate(tag_names)}
    problem_tags_mh = np.zeros((len(problems), len(tag_names)), dtype=np.uint8)
    for i, tags_list in enumerate(problems["tags_norm"].tolist()):
        if not isinstance(tags_list, list):
            continue
        for t in tags_list:
            j = tag_to_j.get(str(t))
            if j is not None:
                problem_tags_mh[i, j] = 1
    problem_tag_counts = problem_tags_mh.sum(axis=1).astype(np.float32)

    # --- Baselines（只用训练窗口统计；不偷看测试窗口） ---
    pop_ac = subs_train[subs_train["ac"] == 1].groupby("problem_id").size()
    if len(pop_ac) == 0:
        pop_ac = subs_train.groupby("problem_id").size()
    pop_count = pop_ac.to_dict()

    popular_ranked = sorted(
        [int(pid) for pid in problems["problem_id"].astype(int).tolist()],
        key=lambda pid: (-int(pop_count.get(int(pid), 0)), int(pid)),
    )
    easy_ranked = sorted(
        [int(pid) for pid in problems["problem_id"].astype(int).tolist()],
        key=lambda pid: (int(pid_to_diff.get(int(pid), diff_median)), -int(pop_count.get(int(pid), 0)), int(pid)),
    )

    # 严格无泄漏：用户画像/偏好只用训练窗口（subs_train）统计得到
    # 这里按 user-problem 聚合后再统计，避免同题多次提交对画像造成不必要放大
    up = subs_train.groupby(["user_id", "problem_id"], as_index=False).agg(
        n_attempts=("submission_id", "count"),
        solved=("ac", "max"),
    )
    up = up.merge(problems[["problem_id", "difficulty_filled"]], on="problem_id", how="left")
    up["difficulty_filled"] = pd.to_numeric(up["difficulty_filled"], errors="coerce").fillna(diff_median).astype(int)
    up["diff_norm"] = up["difficulty_filled"].astype(float) / 10.0

    num = (up["solved"].astype(float) * up["diff_norm"]).groupby(up["user_id"]).sum()
    den = up["diff_norm"].groupby(up["user_id"]).sum()
    level_s = (num / (den + 1e-9)).fillna(0.0).clip(0.0, 1.0)

    # perseverance：训练窗内平均每题提交次数，做 log1p 再按全局 P95 归一化
    avg_attempts_per_problem = (up.groupby("user_id")["n_attempts"].mean()).fillna(0.0).astype(float)
    p95 = float(np.percentile(avg_attempts_per_problem.values, 95)) if len(avg_attempts_per_problem) else 1.0
    denom_p = math.log1p(p95) if p95 > 0 else 1.0
    perseverance_s = (np.log1p(avg_attempts_per_problem) / (denom_p if denom_p > 0 else 1.0)).clip(0.0, 1.0)

    # language preference：训练窗内语言分布（比例）
    lang_counts = subs_train.groupby(["user_id", "language"]).size().reset_index(name="cnt")
    lang_tab = lang_counts.pivot_table(index="user_id", columns="language", values="cnt", fill_value=0)
    lang_tab = lang_tab.div(lang_tab.sum(axis=1).replace(0, 1), axis=0)
    user_lang_pref: dict[int, dict[str, float]] = {}
    for uid, row in lang_tab.iterrows():
        d: dict[str, float] = {}
        for l in lang_names:
            v = float(row.get(l, 0.0))
            if v > 0:
                d[l] = v
        user_lang_pref[int(uid)] = d

    # tag_pref：按训练窗内做过的题（去重到 user-problem）统计其 tags_norm
    pid_to_tags_norm = {
        int(pid): (lst if isinstance(lst, list) else [])
        for pid, lst in zip(problems["problem_id"].astype(int), problems["tags_norm"].tolist())
    }
    tag_rows: list[tuple[int, str]] = []
    for uid, pid in up[["user_id", "problem_id"]].itertuples(index=False):
        for t in pid_to_tags_norm.get(int(pid), []):
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

    # 汇总用户数值画像（推荐打分时填入 level/perseverance）
    user_features = {
        int(uid): {"level": float(level_s.get(uid, 0.0)), "perseverance": float(perseverance_s.get(uid, 0.0))}
        for uid in sorted(set(subs_train["user_id"].astype(int).tolist()))
    }

    # 若某用户无语言历史，则回退到一个全局默认语言（取列顺序第一个）
    global_lang = lang_names[0] if lang_names else ""

    def top_language_for_user(uid: int) -> tuple[str, float]:
        """返回用户最常用语言及其比例；若缺失则返回默认语言与 0。"""
        pref = user_lang_pref.get(uid, {}) or {}
        if not pref:
            return global_lang, 0.0
        best = None
        best_p = -1.0
        for l in lang_names:
            p = float(pref.get(l, 0.0))
            if p > best_p:
                best_p = p
                best = l
        return (best or global_lang), max(0.0, float(best_p))

    def lang_vec_for_choice(chosen: str) -> np.ndarray:
        """把选定语言转换成与 train_samples 一致顺序的 one-hot 向量。"""
        if not lang_cols:
            return np.zeros((0,), dtype=np.float32)
        v = np.zeros((len(lang_cols),), dtype=np.float32)
        if chosen and chosen in lang_names:
            v[lang_names.index(chosen)] = 1.0
        return v

    numeric_idx = {
        "attempt_no": col_to_idx["attempt_no"],
        "difficulty_filled": col_to_idx["difficulty_filled"],
        "level": col_to_idx["level"],
        "perseverance": col_to_idx["perseverance"],
        "lang_match": col_to_idx["lang_match"],
        "tag_match": col_to_idx["tag_match"],
    }
    lang_idx = [col_to_idx[c] for c in lang_cols]
    tag_idx = [col_to_idx[c] for c in tag_cols]

    def recommend_for_user(uid: int) -> list[dict]:
        """
        为单个用户生成两种模型策略的 Top‑K 推荐明细：
        - model_maxprob：按预测 AC 概率排序
        - model_growth：优先成长带（in_growth_band=1），再按离成长带/目标概率最近补齐
        """
        feat = user_features.get(uid, {"level": 0.0, "perseverance": 0.0})
        level = float(feat["level"])
        perseverance = float(feat["perseverance"])
        chosen_lang, chosen_lang_p = top_language_for_user(uid)
        lvec = lang_vec_for_choice(chosen_lang)
        tpref = user_tag_pref.get(uid, {}) or {}
        tag_pref_vec = np.asarray([float(tpref.get(t, 0.0)) for t in tag_names], dtype=np.float32)

        # 候选集合：所有题目中剔除训练窗内已 AC 的题
        solved = solved_map.get(uid, set())
        if solved:
            cand_mask = ~np.isin(problem_ids, np.fromiter(solved, dtype=np.int32))
            cand_pos = np.nonzero(cand_mask)[0]
        else:
            cand_pos = np.arange(len(problem_ids), dtype=np.int32)

        attempt_next = attempt_next_map.get(uid, {})

        # 分块构造特征并打分，避免一次性对全量候选题构造巨大矩阵
        all_pids: list[int] = []
        all_probs: list[float] = []
        all_diffs: list[int] = []

        for start in range(0, len(cand_pos), CHUNK_SIZE):
            pos = cand_pos[start: start + CHUNK_SIZE]
            pids = problem_ids[pos]

            # attempt_no：用户对该题的“下一次提交序号”（没做过则为 1）
            attempt_no = np.fromiter(
                (attempt_next.get(int(pid), 1) for pid in pids), dtype=np.int32, count=len(pids)
            )

            # 构造与训练阶段一致的特征向量（列顺序严格按 feature_cols）
            X = np.zeros((len(pos), len(feature_cols)), dtype=np.float32)
            X[:, numeric_idx["attempt_no"]] = attempt_no
            X[:, numeric_idx["difficulty_filled"]] = problem_diff[pos]
            X[:, numeric_idx["level"]] = level
            X[:, numeric_idx["perseverance"]] = perseverance

            # lang_match：这里用“最常用语言的历史占比”近似表示当前语言匹配度
            X[:, numeric_idx["lang_match"]] = float(chosen_lang_p)
            if tag_idx:
                # tag_match：用用户 tag 偏好分布与题目 tags multi-hot 的加权相似度（再除以题目标签数）
                tm_sum = (problem_tags_mh[pos].astype(np.float32) * tag_pref_vec).sum(axis=1)
                tm_den = np.maximum(1.0, problem_tag_counts[pos])
                X[:, numeric_idx["tag_match"]] = tm_sum / tm_den
            else:
                X[:, numeric_idx["tag_match"]] = 0.0

            # one-hot / multi-hot 稀疏特征填充
            if lang_idx:
                X[:, lang_idx] = lvec
            if tag_idx:
                X[:, tag_idx] = problem_tags_mh[pos].astype(np.float32)

            # 预测 AC 概率
            prob = model.predict_proba(X)[:, 1]
            all_pids.extend([int(x) for x in pids.tolist()])
            all_probs.extend([float(x) for x in prob.tolist()])
            all_diffs.extend([int(x) for x in problem_diff[pos].tolist()])

        pids_arr = np.asarray(all_pids, dtype=np.int32)
        probs_arr = np.asarray(all_probs, dtype=np.float32)
        diffs_arr = np.asarray(all_diffs, dtype=np.int32)

        # 成长带筛选：把概率落在区间内的题标为 in_band
        in_band = (probs_arr >= GROWTH_MIN_P) & (probs_arr <= GROWTH_MAX_P)

        # 带内排序偏好：默认更靠近下限（更具挑战），target_p 设在区间靠近下限的位置
        target_p = float(GROWTH_MIN_P + 0.20 * (GROWTH_MAX_P - GROWTH_MIN_P))

        # 带外补齐：按距离成长带最近优先
        dist_to_band = np.where(
            in_band,
            0.0,
            np.minimum(np.abs(probs_arr - float(GROWTH_MIN_P)), np.abs(probs_arr - float(GROWTH_MAX_P))),
        ).astype(np.float32)
        dist_to_target = np.abs(probs_arr - float(target_p)).astype(np.float32)

        # growth-band first: 带内优先（默认更靠近下限、更有挑战）；带外按“离成长带最近”补齐
        # np.lexsort 的“最后一个 key 为主键”：这里依次优先级为
        # 1) (~in_band)：带内优先
        # 2) dist_to_band：带外时离区间越近越优先
        # 3) dist_to_target：带内时离 target_p 越近越优先（更靠近下限）
        # 4) -difficulty：同条件下优先更难的题
        # 5) prob：再同条件下优先更低概率（更挑战）
        # 6) pid：最终打散，保证确定性
        order_growth = np.lexsort(
            (
                pids_arr.astype(np.int32),
                probs_arr.astype(np.float32),  # harder-first when tie
                -diffs_arr.astype(np.int32),  # harder-first when tie
                dist_to_target,
                dist_to_band,
                (~in_band).astype(np.int8),
            )
        )
        chosen_growth = order_growth[:MAX_K].tolist()

        # max-prob top-k
        chosen_max = np.argsort(probs_arr)[::-1][:MAX_K].tolist()

        def rows_from_idx(chosen_idx: list[int], *, strategy: str) -> list[dict]:
            """将候选数组索引转换成可落 CSV 的推荐明细行。"""
            recs: list[dict] = []
            for rank, i in enumerate(chosen_idx, start=1):
                recs.append(
                    {
                        "strategy": strategy,
                        "user_id": uid,
                        "rank": rank,
                        "problem_id": int(pids_arr[i]),
                        "p_ac": float(probs_arr[i]),
                        "difficulty": int(diffs_arr[i]),
                        "in_growth_band": int(in_band[i]),
                    }
                )
            return recs

        return rows_from_idx(chosen_max, strategy="model_maxprob") + rows_from_idx(
            chosen_growth, strategy="model_growth"
        )

    def baseline_recommend(uid: int, ranked: list[int], *, strategy: str) -> list[dict]:
        """基于给定全局排序（popular/easy）生成该用户 Top‑K，剔除已解题。"""
        solved = solved_map.get(uid, set())
        out: list[dict] = []
        for pid in ranked:
            if pid in solved:
                continue
            out.append(
                {
                    "strategy": strategy,
                    "user_id": uid,
                    "rank": len(out) + 1,
                    "problem_id": int(pid),
                    "p_ac": float("nan"),
                    "difficulty": int(pid_to_diff.get(int(pid), diff_median)),
                    "in_growth_band": 0,
                }
            )
            if len(out) >= MAX_K:
                break
        return out

    def random_recommend(uid: int) -> list[dict]:
        """随机策略：从候选题中无放回抽样 Top‑K（剔除已解题）。"""
        solved = solved_map.get(uid, set())
        cand = [int(pid) for pid in problems["problem_id"].astype(int).tolist() if int(pid) not in solved]
        if not cand:
            return []
        if len(cand) <= MAX_K:
            picks = cand
        else:
            picks = rng.choice(np.asarray(cand, dtype=np.int32), size=MAX_K, replace=False).astype(int).tolist()
        out: list[dict] = []
        for rank, pid in enumerate(picks, start=1):
            out.append(
                {
                    "strategy": "random",
                    "user_id": uid,
                    "rank": rank,
                    "problem_id": int(pid),
                    "p_ac": float("nan"),
                    "difficulty": int(pid_to_diff.get(int(pid), diff_median)),
                    "in_growth_band": 0,
                }
            )
        return out

    # 为所有用户生成推荐，并按策略收集推荐列表用于指标计算
    users = sorted(user_features.keys())
    reco_compare_rows: list[dict] = []
    reco_growth_rows: list[dict] = []
    rec_pids_by_strategy: dict[str, dict[int, list[int]]] = {s: {} for s in STRATEGIES}

    for i, uid in enumerate(users, start=1):
        uid = int(uid)
        model_rows = recommend_for_user(uid)
        pop_rows = baseline_recommend(uid, popular_ranked, strategy="popular_train")
        easy_rows = baseline_recommend(uid, easy_ranked, strategy="easy_first")
        rand_rows = random_recommend(uid)

        all_rows = model_rows + pop_rows + easy_rows + rand_rows
        reco_compare_rows.extend(all_rows)

        # keep old output: model_growth only
        for r in model_rows:
            if r["strategy"] == "model_growth":
                reco_growth_rows.append(
                    {
                        "user_id": r["user_id"],
                        "rank": r["rank"],
                        "problem_id": r["problem_id"],
                        "p_ac": r["p_ac"],
                        "difficulty": r["difficulty"],
                        "in_growth_band": r["in_growth_band"],
                    }
                )

        # index by strategy for metrics
        for s in STRATEGIES:
            pids = [int(r["problem_id"]) for r in all_rows if r["strategy"] == s]
            rec_pids_by_strategy[s][uid] = pids

        if i % 50 == 0:
            print(f"scored users: {i}/{len(users)}")

    # 兼容旧输出：仅保存 model_growth
    reco_df = pd.DataFrame(reco_growth_rows).sort_values(["user_id", "rank"])
    reco_df.to_csv(OUT_RECO, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_RECO} rows={len(reco_df)} users={reco_df['user_id'].nunique()}")

    # 多策略对比输出：用于分析不同策略的推荐差异
    reco_cmp_df = pd.DataFrame(reco_compare_rows).sort_values(["strategy", "user_id", "rank"])
    reco_cmp_df.to_csv(OUT_RECO_COMPARE, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_RECO_COMPARE} rows={len(reco_cmp_df)}")

    # metrics
    metric_rows: list[dict] = []
    for strategy in STRATEGIES:
        for k in KS:
            # all：所有训练窗用户（即参与推荐的 users）
            hits = []
            precs = []
            # active：测试窗内有提交的用户（更接近线上实际）
            hits_active = []
            precs_active = []
            # users_with_test_ac：测试窗内至少 AC 过 1 题的用户，才能定义 recall/ndcg
            recalls_ac = []
            ndcgs_ac = []

            for uid in users:
                uid = int(uid)
                rec_pids = rec_pids_by_strategy[strategy].get(uid, [])
                topk = rec_pids[: int(k)]
                rec_set = set(topk)
                gt = test_ac_map.get(uid, set())
                inter = len(rec_set & gt)
                hit = 1 if inter > 0 else 0
                prec = inter / float(k)
                hits.append(hit)
                precs.append(prec)

                if uid in users_with_test:
                    hits_active.append(hit)
                    precs_active.append(prec)

                if uid in users_with_test_ac and gt:
                    recalls_ac.append(inter / float(len(gt)))
                    ndcgs_ac.append(ndcg_at_k(topk, gt, int(k)))

            metric_rows.append(
                {
                    "strategy": strategy,
                    "k": int(k),
                    "hit_at_k_all": float(np.mean(hits)) if hits else 0.0,
                    "precision_at_k_all": float(np.mean(precs)) if precs else 0.0,
                    "hit_at_k_active": float(np.mean(hits_active)) if hits_active else 0.0,
                    "precision_at_k_active": float(np.mean(precs_active)) if precs_active else 0.0,
                    "recall_at_k_ac_users": float(np.mean(recalls_ac)) if recalls_ac else 0.0,
                    "ndcg_at_k_ac_users": float(np.mean(ndcgs_ac)) if ndcgs_ac else 0.0,
                    "users_all": int(len(users)),
                    "users_active": int(len(users_with_test)),
                    "users_with_test_ac": int(len(users_with_test_ac)),
                    "growth_band": f"[{GROWTH_MIN_P},{GROWTH_MAX_P}]",
                    "cutoff_submission_id": int(cutoff_id),
                    "random_seed": int(RANDOM_SEED),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["strategy", "k"])
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_METRICS}")

    # Hit@K curve（active users）
    plt.figure()
    for strategy in STRATEGIES:
        sub = metrics_df[metrics_df["strategy"] == strategy].sort_values("k")
        plt.plot(sub["k"], sub["hit_at_k_active"], marker="o", label=strategy)
    plt.title("Hit@K 随 K 变化（策略对比；命中=测试窗口内是否AC）")
    plt.xlabel("K")
    plt.ylabel("Hit@K (active users)")
    plt.xticks(list(KS))
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_HITK, dpi=200)
    plt.close()

    # Coverage / concentration：Top‑K 推荐的覆盖率与集中度（以 model_growth 为例）
    topk = reco_df[reco_df["rank"] <= MAX_K]
    total = len(topk)
    uniq = int(topk["problem_id"].nunique())
    coverage = uniq / float(total) if total else 0.0
    freq = topk["problem_id"].value_counts().head(20)
    plt.figure(figsize=(10, 4))
    plt.bar([str(x) for x in freq.index.tolist()], freq.values.tolist())
    plt.title(f"推荐集中度（Top20 题目），覆盖率={coverage:.3f}")
    plt.xlabel("problem_id")
    plt.ylabel("被推荐次数")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_COVERAGE, dpi=200)
    plt.close()

    # Case：选一个 active user 画推荐难度分布（用于直观检查“是否过难/过易”）
    case_uid = None
    for uid in users:
        if uid in users_with_test:
            case_uid = uid
            break
    if case_uid is None:
        case_uid = users[0] if users else 0

    case = reco_df[(reco_df["user_id"] == case_uid) & (reco_df["rank"] <= MAX_K)]
    plt.figure()
    plt.hist(case["difficulty"].astype(int), bins=np.arange(0.5, 10.6, 1), edgecolor="black")
    plt.title(f"推荐题目难度分布（user_id={case_uid}）")
    plt.xlabel("难度（1-10）")
    plt.ylabel("数量")
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(FIG_CASE_DIFF, dpi=200)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
