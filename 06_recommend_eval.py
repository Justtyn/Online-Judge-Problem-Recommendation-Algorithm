import json
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
STUDENTS = "CleanData/students_derived.csv"
TAGS = "CleanData/tags.csv"
TRAIN_SAMPLES = "FeatureData/train_samples.csv"

OUT_RECO = "Reports/recommendations_topk.csv"
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


def main() -> int:
    np.random.seed(RANDOM_SEED)

    ensure_dir(OUT_RECO)
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

    students = pd.read_csv(STUDENTS)
    students["user_id"] = pd.to_numeric(students["user_id"], errors="coerce").astype(int)
    students["level"] = pd.to_numeric(students["level"], errors="coerce").fillna(0.0).astype(float)
    students["perseverance"] = (
        pd.to_numeric(students["perseverance"], errors="coerce").fillna(0.0).astype(float)
    )
    students["lang_pref_dict"] = students["lang_pref"].apply(parse_json_dict)
    students["tag_pref_dict"] = students["tag_pref"].apply(parse_json_dict)
    user_features = students.set_index("user_id")[["level", "perseverance"]].to_dict("index")
    user_lang_pref = students.set_index("user_id")["lang_pref_dict"].to_dict()
    user_tag_pref = students.set_index("user_id")["tag_pref_dict"].to_dict()

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
            pos = cand_pos[start : start + CHUNK_SIZE]
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
        band_idx = np.nonzero(in_band)[0]

        chosen: list[int] = []
        chosen_set: set[int] = set()

        if len(band_idx):
            take = band_idx[np.argsort(probs_arr[band_idx])[::-1]].tolist()
            for i in take:
                pid = int(pids_arr[i])
                if pid in chosen_set:
                    continue
                chosen.append(i)
                chosen_set.add(pid)
                if len(chosen) >= MAX_K:
                    break

        if len(chosen) < MAX_K:
            all_order = np.argsort(probs_arr)[::-1]
            for i in all_order.tolist():
                pid = int(pids_arr[i])
                if pid in chosen_set:
                    continue
                chosen.append(i)
                chosen_set.add(pid)
                if len(chosen) >= MAX_K:
                    break

        recs: list[dict] = []
        for rank, i in enumerate(chosen, start=1):
            recs.append(
                {
                    "user_id": uid,
                    "rank": rank,
                    "problem_id": int(pids_arr[i]),
                    "p_ac": float(probs_arr[i]),
                    "difficulty": int(diffs_arr[i]),
                    "in_growth_band": int(in_band[i]),
                }
            )
        return recs

    users = sorted(students["user_id"].astype(int).tolist())
    all_recs: list[dict] = []
    users_with_test = {int(x) for x in subs_test["user_id"].unique().tolist()}

    for i, uid in enumerate(users, start=1):
        all_recs.extend(recommend_for_user(int(uid)))
        if i % 50 == 0:
            print(f"scored users: {i}/{len(users)}")

    reco_df = pd.DataFrame(all_recs).sort_values(["user_id", "rank"])
    reco_df.to_csv(OUT_RECO, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_RECO} rows={len(reco_df)} users={reco_df['user_id'].nunique()}")

    # metrics
    rows = []
    for k in KS:
        hits = []
        precs = []
        hits_active = []
        precs_active = []

        for uid in users:
            rec_k = reco_df[(reco_df["user_id"] == uid) & (reco_df["rank"] <= k)][
                "problem_id"
            ].astype(int)
            rec_set = set(rec_k.tolist())
            gt = test_ac_map.get(int(uid), set())
            inter = len(rec_set & gt)
            hit = 1 if inter > 0 else 0
            prec = inter / float(k)
            hits.append(hit)
            precs.append(prec)

            if uid in users_with_test:
                hits_active.append(hit)
                precs_active.append(prec)

        rows.append(
            {
                "k": int(k),
                "hit_at_k_all": float(np.mean(hits)) if hits else 0.0,
                "precision_at_k_all": float(np.mean(precs)) if precs else 0.0,
                "hit_at_k_active": float(np.mean(hits_active)) if hits_active else 0.0,
                "precision_at_k_active": float(np.mean(precs_active)) if precs_active else 0.0,
                "users_all": int(len(users)),
                "users_active": int(len(users_with_test)),
                "growth_band": f"[{GROWTH_MIN_P},{GROWTH_MAX_P}]",
                "cutoff_submission_id": int(cutoff_id),
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_METRICS}")

    # Hit@K curve
    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["hit_at_k_active"], marker="o", label="Hit@K (active users)")
    plt.plot(metrics_df["k"], metrics_df["hit_at_k_all"], marker="o", label="Hit@K (all users)")
    plt.title("Hit@K 随 K 变化（命中=测试窗口内是否AC）")
    plt.xlabel("K")
    plt.ylabel("Hit@K")
    plt.xticks(list(KS))
    plt.grid(True, alpha=0.3)
    plt.legend(["Hit@K（活跃用户）", "Hit@K（全量用户）"])
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
