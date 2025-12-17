import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUBMISSIONS = "CleanData/submissions.csv"
PROBLEMS = "CleanData/problems.csv"
TAGS = "CleanData/tags.csv"
TRAIN_SAMPLES = "FeatureData/train_samples.csv"

OUT_RECO = "Reports/recommendations_topk.csv"
OUT_RECO_COMPARE = "Reports/recommendations_topk_compare.csv"
OUT_METRICS = "Reports/reco_metrics.csv"
FIG_HITK = "Reports/fig_hitk_curve.png"
FIG_CASE_DIFF = "Reports/fig_reco_difficulty_hist.png"
FIG_COVERAGE = "Reports/fig_reco_coverage.png"

TIME_SPLIT = 0.8
KS = (1, 3, 5, 10)
MAX_K = max(KS)

GROWTH_MIN_P = 0.4
GROWTH_MAX_P = 0.7

CHUNK_SIZE = 2048
RANDOM_SEED = 42

STRATEGIES = (
    "model_maxprob",
    "model_growth",
    "popular_train",
    "easy_first",
    "random",
)


def setup_cn_font() -> None:
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


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ndcg_at_k(rec_pids: list[int], gt: set[int], k: int) -> float:
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

    ensure_dir(OUT_RECO)
    ensure_dir(OUT_RECO_COMPARE)
    ensure_dir(OUT_METRICS)
    ensure_dir(FIG_HITK)
    ensure_dir(FIG_CASE_DIFF)
    ensure_dir(FIG_COVERAGE)

    df = pd.read_csv(TRAIN_SAMPLES)
    df["submission_id"] = pd.to_numeric(df["submission_id"], errors="coerce").astype(int)

    feature_cols = [
        c for c in df.columns if c not in {"ac", "submission_id", "user_id", "problem_id"}
    ]
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}

    required = {"attempt_no", "difficulty_filled", "level", "perseverance", "lang_match", "tag_match"}
    missing = sorted(required - set(feature_cols))
    if missing:
        raise SystemExit(f"train_samples 缺少必要特征列：{missing}")

    lang_cols = [c for c in feature_cols if c.startswith("lang_")]
    tag_cols = [c for c in feature_cols if c.startswith("tag_")]
    lang_names = [c.removeprefix("lang_") for c in lang_cols]
    tag_names = [c.removeprefix("tag_") for c in tag_cols]

    order = np.argsort(df["submission_id"].values)
    df = df.iloc[order].reset_index(drop=True)
    split = int(len(df) * TIME_SPLIT)
    cutoff_id = int(df["submission_id"].iloc[split - 1])

    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["ac"].astype(int).values
    # test_df currently unused for recommendation metrics (Hit@K uses CleanData/submissions.csv time window)

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(X_train.to_numpy(dtype=np.float32), y_train)

    subs = pd.read_csv(SUBMISSIONS)
    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    subs_train = subs[subs["submission_id"] <= cutoff_id].copy()
    subs_test = subs[subs["submission_id"] > cutoff_id].copy()

    users_with_test = {int(x) for x in subs_test["user_id"].unique().tolist()}

    solved_before = (
        subs_train[subs_train["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
    )
    solved_map: dict[int, set[int]] = {}
    for uid, g in solved_before.groupby("user_id"):
        solved_map[int(uid)] = set(g["problem_id"].astype(int).tolist())

    attempts_before = subs_train.groupby(["user_id", "problem_id"]).size().reset_index(name="n")
    attempt_next_map: dict[int, dict[int, int]] = {}
    for uid, g in attempts_before.groupby("user_id"):
        attempt_next_map[int(uid)] = {
            int(pid): int(n) + 1 for pid, n in zip(g["problem_id"], g["n"], strict=False)
        }

    test_ac = subs_test[subs_test["ac"] == 1][["user_id", "problem_id"]].drop_duplicates()
    test_ac_map: dict[int, set[int]] = {}
    for uid, g in test_ac.groupby("user_id"):
        test_ac_map[int(uid)] = set(g["problem_id"].astype(int).tolist())
    users_with_test_ac = set(test_ac_map.keys())

    problems = pd.read_csv(PROBLEMS)
    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_norm"] = problems["tags"].apply(parse_json_list)

    tag_whitelist = set(pd.read_csv(TAGS)["tag_name"].astype(str).tolist())
    problems["tags_norm"] = problems["tags_norm"].apply(lambda lst: [t for t in lst if t in tag_whitelist])

    problem_ids = problems["problem_id"].to_numpy(dtype=np.int32)
    problem_diff = problems["difficulty_filled"].to_numpy(dtype=np.int32)
    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int), strict=False))

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

    # --- Baselines (train-only; no peeking at test) ---
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

    avg_attempts_per_problem = (up.groupby("user_id")["n_attempts"].mean()).fillna(0.0).astype(float)
    p95 = float(np.percentile(avg_attempts_per_problem.values, 95)) if len(avg_attempts_per_problem) else 1.0
    denom_p = math.log1p(p95) if p95 > 0 else 1.0
    perseverance_s = (np.log1p(avg_attempts_per_problem) / (denom_p if denom_p > 0 else 1.0)).clip(0.0, 1.0)

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

    user_features = {
        int(uid): {"level": float(level_s.get(uid, 0.0)), "perseverance": float(perseverance_s.get(uid, 0.0))}
        for uid in sorted(set(subs_train["user_id"].astype(int).tolist()))
    }

    global_lang = lang_names[0] if lang_names else ""

    def top_language_for_user(uid: int) -> tuple[str, float]:
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
        feat = user_features.get(uid, {"level": 0.0, "perseverance": 0.0})
        level = float(feat["level"])
        perseverance = float(feat["perseverance"])
        chosen_lang, chosen_lang_p = top_language_for_user(uid)
        lvec = lang_vec_for_choice(chosen_lang)
        tpref = user_tag_pref.get(uid, {}) or {}
        tag_pref_vec = np.asarray([float(tpref.get(t, 0.0)) for t in tag_names], dtype=np.float32)

        solved = solved_map.get(uid, set())
        if solved:
            cand_mask = ~np.isin(problem_ids, np.fromiter(solved, dtype=np.int32))
            cand_pos = np.nonzero(cand_mask)[0]
        else:
            cand_pos = np.arange(len(problem_ids), dtype=np.int32)

        attempt_next = attempt_next_map.get(uid, {})

        all_pids: list[int] = []
        all_probs: list[float] = []
        all_diffs: list[int] = []

        for start in range(0, len(cand_pos), CHUNK_SIZE):
            pos = cand_pos[start: start + CHUNK_SIZE]
            pids = problem_ids[pos]

            attempt_no = np.fromiter(
                (attempt_next.get(int(pid), 1) for pid in pids), dtype=np.int32, count=len(pids)
            )

            X = np.zeros((len(pos), len(feature_cols)), dtype=np.float32)
            X[:, numeric_idx["attempt_no"]] = attempt_no
            X[:, numeric_idx["difficulty_filled"]] = problem_diff[pos]
            X[:, numeric_idx["level"]] = level
            X[:, numeric_idx["perseverance"]] = perseverance
            X[:, numeric_idx["lang_match"]] = float(chosen_lang_p)
            if tag_idx:
                tm_sum = (problem_tags_mh[pos].astype(np.float32) * tag_pref_vec).sum(axis=1)
                tm_den = np.maximum(1.0, problem_tag_counts[pos])
                X[:, numeric_idx["tag_match"]] = tm_sum / tm_den
            else:
                X[:, numeric_idx["tag_match"]] = 0.0
            if lang_idx:
                X[:, lang_idx] = lvec
            if tag_idx:
                X[:, tag_idx] = problem_tags_mh[pos].astype(np.float32)

            prob = model.predict_proba(X)[:, 1]
            all_pids.extend([int(x) for x in pids.tolist()])
            all_probs.extend([float(x) for x in prob.tolist()])
            all_diffs.extend([int(x) for x in problem_diff[pos].tolist()])

        pids_arr = np.asarray(all_pids, dtype=np.int32)
        probs_arr = np.asarray(all_probs, dtype=np.float32)
        diffs_arr = np.asarray(all_diffs, dtype=np.int32)

        in_band = (probs_arr >= GROWTH_MIN_P) & (probs_arr <= GROWTH_MAX_P)
        target_p = float(GROWTH_MIN_P + 0.20 * (GROWTH_MAX_P - GROWTH_MIN_P))
        dist_to_band = np.where(
            in_band,
            0.0,
            np.minimum(np.abs(probs_arr - float(GROWTH_MIN_P)), np.abs(probs_arr - float(GROWTH_MAX_P))),
        ).astype(np.float32)
        dist_to_target = np.abs(probs_arr - float(target_p)).astype(np.float32)

        # growth-band first: 带内优先（默认更靠近下限、更有挑战）；带外按“离成长带最近”补齐
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

    reco_df = pd.DataFrame(reco_growth_rows).sort_values(["user_id", "rank"])
    reco_df.to_csv(OUT_RECO, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_RECO} rows={len(reco_df)} users={reco_df['user_id'].nunique()}")

    reco_cmp_df = pd.DataFrame(reco_compare_rows).sort_values(["strategy", "user_id", "rank"])
    reco_cmp_df.to_csv(OUT_RECO_COMPARE, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_RECO_COMPARE} rows={len(reco_cmp_df)}")

    # metrics
    metric_rows: list[dict] = []
    for strategy in STRATEGIES:
        for k in KS:
            hits = []
            precs = []
            hits_active = []
            precs_active = []
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

    # Hit@K curve
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

    # Coverage / concentration
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

    # Case: difficulty histogram for one active user (if any)
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
