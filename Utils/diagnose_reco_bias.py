"""
Utils/diagnose_reco_bias.py

用途
- 诊断“推荐结果偏简单/被难度 1 刷屏”等问题，并提供可量化的证据与可失败的校验 flags。
- 支持两种模式：
  - 单用户：输出文本报告到 `Reports/diag/`，并可选输出图表到 `Reports/fig/`
  - 扫描：批量扫描多用户，输出汇总 CSV（便于批处理或接 CI）

典型用法
- 单用户：`python Utils/diagnose_reco_bias.py --user-id 104 --plot`
- 扫描：`python Utils/diagnose_reco_bias.py --scan --max-users 300 --fail-if-any`

说明
- 该脚本复用 `WebApp.server.Recommender` 的推理与画像逻辑，以保证诊断与 Web 端行为一致。
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from WebApp.server import Recommender

REPORTS_DIR = ROOT / "Reports"
REPORTS_FIG_DIR = REPORTS_DIR / "fig"
REPORTS_DIAG_DIR = REPORTS_DIR / "diag"


@dataclass(frozen=True)
class Diagnosis:
    user_id: int
    cutoff_pct: float
    cutoff_submission_id: int
    hist_submissions: int
    hist_solved: int
    hist_attempted_unique: int
    hist_solved_median_diff: float
    hist_attempted_median_diff: float
    reco_k: int
    reco_median_diff: float
    reco_easy_share: float
    reco_in_band_share: float
    reco_score_std: float
    flags: list[str]


def _parse_user_list(s: str) -> list[int]:
    """解析逗号分隔的 user_id 列表字符串。"""
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def _safe_quantile(x: np.ndarray, q: float) -> float:
    """对数组做安全分位数计算（自动过滤 NaN/inf；空则返回 nan）。"""
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _median_or_nan(xs: list[int]) -> float:
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float32)))


def _difficulty_for_pids(reco: Recommender, pids: np.ndarray) -> np.ndarray:
    out = np.full((len(pids),), 5, dtype=np.int32)
    for j, pid in enumerate(pids.tolist()):
        i = reco.pid_to_i.get(int(pid))
        if i is not None:
            out[j] = int(reco.problem_diff[i])
    return out


def _compute_user_scores(
        reco: Recommender,
        *,
        user_id: int,
        cutoff_pct: float,
        min_p: float,
        max_p: float,
) -> dict:
    """
    计算某用户在 cutoff 切片点下的候选集分数、成长带命中与辅助向量。

    返回的字典仅用于诊断（含候选 mask / score 分布），不作为稳定 API。
    """
    user_id = int(user_id)
    user_df = reco._subs_by_user.get(user_id)
    if user_df is None or user_df.empty:
        raise RuntimeError(f"user_id={user_id} 不存在或无 submissions")

    cutoff_id = reco._user_cutoff_id(user_id, cutoff_pct)
    level, perseverance, lang_pref, tag_pref, solved, attempt_next_map = reco._profile_from_history(user_df, cutoff_id)

    chosen_lang = ""
    if reco.lang_names:
        chosen_lang = max(reco.lang_names, key=lambda l: float(lang_pref.get(l, 0.0)))

    attempt_no_vec = np.ones((len(reco.problem_ids),), dtype=np.float32)
    for pid, next_no in attempt_next_map.items():
        i = reco.pid_to_i.get(int(pid))
        if i is not None:
            attempt_no_vec[i] = float(max(1, min(10, int(next_no))))

    solved_mask = (
        np.isin(reco.problem_ids, np.fromiter((int(x) for x in solved), dtype=np.int32))
        if solved
        else np.zeros((len(reco.problem_ids),), dtype=bool)
    )
    candidate_mask = ~solved_mask

    tag_pref_vec = np.asarray([float(tag_pref.get(t, 0.0)) for t in reco.tag_names], dtype=np.float32)
    tm_sum = (reco.problem_tags_mh.astype(np.float32) * tag_pref_vec).sum(axis=1)
    tm_den = np.maximum(1.0, reco.problem_tag_counts)
    tag_match = (tm_sum / tm_den).astype(np.float32)

    X = np.zeros((len(reco.problem_ids), len(reco.feature_cols)), dtype=np.float32)
    X[:, reco.numeric_idx["attempt_no"]] = attempt_no_vec
    X[:, reco.numeric_idx["difficulty_filled"]] = reco.problem_diff.astype(np.float32)
    X[:, reco.numeric_idx["level"]] = float(level)
    X[:, reco.numeric_idx["perseverance"]] = float(perseverance)
    X[:, reco.numeric_idx["tag_match"]] = tag_match
    X[:, reco.tag_idx] = reco.problem_tags_mh.astype(np.float32)

    lang_match_val = float(lang_pref.get(chosen_lang, 0.0)) if chosen_lang else 0.0
    X[:, reco.numeric_idx["lang_match"]] = lang_match_val
    if reco.lang_idx and chosen_lang in reco.lang_names:
        X[:, reco.lang_idx] = 0.0
        X[:, reco.lang_idx[reco.lang_names.index(chosen_lang)]] = 1.0

    score = reco.model.predict_proba(X)[:, 1].astype(np.float32)
    min_p = float(max(0.0, min(1.0, min_p)))
    max_p = float(max(0.0, min(1.0, max_p)))
    if max_p < min_p:
        min_p, max_p = max_p, min_p
    in_band = (score >= min_p) & (score <= max_p)

    return {
        "user_df": user_df,
        "cutoff_id": int(cutoff_id),
        "level": float(level),
        "perseverance": float(perseverance),
        "chosen_lang": str(chosen_lang),
        "candidate_mask": candidate_mask,
        "score": score,
        "in_band": in_band,
        "tag_match": tag_match,
    }


def _diagnose_one(
        reco: Recommender,
        *,
        user_id: int,
        cutoff_pct: float,
        k: int,
        min_p: float,
        max_p: float,
        plot: bool,
        out_dir: Path,
) -> tuple[Diagnosis, str, list[Path]]:
    # out_dir：文本/表格诊断输出目录（默认 Reports/diag）
    out_dir.mkdir(parents=True, exist_ok=True)
    # 图表统一输出到 Reports/fig（便于 WebApp 统一展示与引用）
    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    debug = _compute_user_scores(reco, user_id=user_id, cutoff_pct=cutoff_pct, min_p=min_p, max_p=max_p)
    user_df: pd.DataFrame = debug["user_df"]
    cutoff_id = int(debug["cutoff_id"])
    score: np.ndarray = debug["score"]
    in_band: np.ndarray = debug["in_band"]
    candidate_mask: np.ndarray = debug["candidate_mask"]

    meta, rec_rows = reco.recommend_for_user_history(
        user_id=int(user_id),
        cutoff_pct=float(cutoff_pct),
        k=int(k),
        min_p=float(min_p),
        max_p=float(max_p),
    )

    hist = user_df[user_df["submission_id"] <= int(cutoff_id)].copy()
    hist_submissions = int(len(hist))
    solved = set(hist.loc[hist["ac"] == 1, "problem_id"].astype(int).tolist()) if not hist.empty else set()
    attempted_unique = set(hist["problem_id"].astype(int).tolist()) if not hist.empty else set()

    hist_solved_diff = _difficulty_for_pids(reco, np.fromiter(solved, dtype=np.int32)) if solved else np.array([])
    hist_attempted_diff = (
        _difficulty_for_pids(reco, np.fromiter(attempted_unique, dtype=np.int32)) if attempted_unique else np.array([])
    )

    rec_diffs = [int(r["difficulty"]) for r in rec_rows]
    rec_in_band = [int(r["in_growth_band"]) for r in rec_rows]
    rec_scores = [float(r["p_ac"]) for r in rec_rows]

    reco_median_diff = _median_or_nan(rec_diffs)
    reco_easy_share = float(np.mean([1.0 if int(d) <= 2 else 0.0 for d in rec_diffs])) if rec_diffs else 0.0
    reco_in_band_share = float(np.mean([1.0 if int(x) == 1 else 0.0 for x in rec_in_band])) if rec_in_band else 0.0
    reco_score_std = float(np.std(np.asarray(rec_scores, dtype=np.float32))) if rec_scores else float("nan")

    hist_solved_median = float(np.median(hist_solved_diff)) if len(hist_solved_diff) else float("nan")
    hist_attempted_median = float(np.median(hist_attempted_diff)) if len(hist_attempted_diff) else float("nan")

    flags: list[str] = []
    if (
            len(hist_solved_diff) >= 10
            and math.isfinite(hist_solved_median)
            and hist_solved_median >= 4.0
            and len(rec_diffs) >= max(5, int(k))
            and reco_median_diff <= 2.0
            and reco_easy_share >= 0.8
    ):
        flags.append("easy_bias_vs_history")
    if math.isfinite(reco_score_std) and reco_score_std < 1e-4:
        flags.append("score_plateau_topk")
    if reco_in_band_share < 0.5:
        flags.append("low_in_band_share_topk")

    # candidate-by-difficulty stats
    cand_idx = np.where(candidate_mask)[0]
    cand_diff = reco.problem_diff[cand_idx]
    cand_score = score[cand_idx]
    cand_in_band = in_band[cand_idx]

    rows = []
    for d in range(1, 11):
        m = cand_diff == d
        if not np.any(m):
            continue
        s = cand_score[m]
        ib = cand_in_band[m]
        rows.append(
            {
                "difficulty": int(d),
                "n_candidates": int(np.sum(m)),
                "p_min": float(np.min(s)),
                "p_q25": _safe_quantile(s, 0.25),
                "p_med": _safe_quantile(s, 0.50),
                "p_q75": _safe_quantile(s, 0.75),
                "p_max": float(np.max(s)),
                "n_in_band": int(np.sum(ib)),
            }
        )
    cand_stats = pd.DataFrame(rows).sort_values("difficulty")

    report_lines: list[str] = []
    report_lines.append(f"user_id={int(user_id)} cutoff_pct={float(cutoff_pct):.2f} cutoff_submission_id={cutoff_id}")
    report_lines.append(
        f"hist_submissions={hist_submissions} hist_attempted_unique={len(attempted_unique)} hist_solved={len(solved)}"
    )
    report_lines.append(
        f"hist_attempted_median_diff={hist_attempted_median:.2f} hist_solved_median_diff={hist_solved_median:.2f}"
    )
    report_lines.append(
        f"reco_k={int(k)} reco_median_diff={reco_median_diff:.2f} reco_easy_share(<=2)={reco_easy_share:.2%} "
        f"reco_in_band_share={reco_in_band_share:.2%} reco_score_std={reco_score_std:.6f}"
    )
    report_lines.append(f"flags={','.join(flags) if flags else '(none)'}")
    report_lines.append("")
    report_lines.append("Top-K:")
    for r in rec_rows:
        report_lines.append(
            f"  rank={int(r['rank']):02d} pid={int(r['problem_id'])} diff={int(r['difficulty'])} "
            f"p_ac={float(r['p_ac']):.3f} in_band={int(r['in_growth_band'])} tags={r.get('tags', '')}"
        )
    report_lines.append("")
    report_lines.append("Candidate P(AC) stats by difficulty (candidates=not solved before cutoff):")
    if cand_stats.empty:
        report_lines.append("  (empty)")
    else:
        report_lines.append(
            "  diff\tN\tin_band\tp_min\tp_q25\tp_med\tp_q75\tp_max"
        )
        for _, row in cand_stats.iterrows():
            report_lines.append(
                f"  {int(row['difficulty'])}\t{int(row['n_candidates'])}\t{int(row['n_in_band'])}\t"
                f"{row['p_min']:.3f}\t{row['p_q25']:.3f}\t{row['p_med']:.3f}\t{row['p_q75']:.3f}\t{row['p_max']:.3f}"
            )

    explanation = []
    explanation.append("解释：")
    explanation.append("- 旧版会在成长带内按 P(AC) 从高到低选题，容易被“难度1但预测≈0.69”的题占满。")
    explanation.append("- 现在 Web 端已改为：带内优先靠近下限（更有挑战），带外按“离成长带最近”补齐。")
    if not cand_stats.empty:
        top1 = cand_stats.iloc[0]
        top_last = cand_stats.iloc[-1]
        explanation.append(
            f"- 该用户候选集中：最低难度 diff={int(top1['difficulty'])} 的 p_med≈{float(top1['p_med']):.3f}；"
            f"最高 diff={int(top_last['difficulty'])} 的 p_med≈{float(top_last['p_med']):.3f}。"
        )
    report = "\n".join(report_lines + [""] + explanation) + "\n"

    fig_paths: list[Path] = []
    if plot:
        # 1) 推荐难度直方图
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.asarray(rec_diffs, dtype=np.int32), bins=np.arange(0.5, 10.6, 1), edgecolor="white", alpha=0.9)
        ax.set_title(f"Top-{int(k)} 推荐难度分布 (user={int(user_id)})")
        ax.set_xlabel("difficulty (1-10)")
        ax.set_ylabel("count")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25, linestyle="--")
        p1 = REPORTS_FIG_DIR / f"fig_diag_user_{int(user_id)}_reco_diff_hist.png"
        fig.savefig(p1, dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(p1)

        # 2) 候选集：按难度的 P(AC) 中位数/四分位区间
        if not cand_stats.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            xs = cand_stats["difficulty"].astype(int).to_numpy()
            med = cand_stats["p_med"].astype(float).to_numpy()
            q25 = cand_stats["p_q25"].astype(float).to_numpy()
            q75 = cand_stats["p_q75"].astype(float).to_numpy()
            ax.plot(xs, med, marker="o", lw=2, label="median P(AC)")
            ax.fill_between(xs, q25, q75, alpha=0.20, label="IQR (Q25-Q75)")
            ax.axhline(float(min_p), color="#ef4444", lw=1.5, linestyle="--", label="min_p")
            ax.axhline(float(max_p), color="#f59e0b", lw=1.5, linestyle="--", label="max_p")
            ax.set_title(f"候选集预测 P(AC) vs difficulty (user={int(user_id)})")
            ax.set_xlabel("difficulty (1-10)")
            ax.set_ylabel("P(AC)")
            ax.set_xticks(range(1, 11))
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(frameon=False, ncols=3, loc="upper right")
            p2 = REPORTS_FIG_DIR / f"fig_diag_user_{int(user_id)}_candidate_p_by_diff.png"
            fig.savefig(p2, dpi=180, bbox_inches="tight")
            plt.close(fig)
            fig_paths.append(p2)

    diag = Diagnosis(
        user_id=int(user_id),
        cutoff_pct=float(cutoff_pct),
        cutoff_submission_id=int(cutoff_id),
        hist_submissions=int(hist_submissions),
        hist_solved=int(len(solved)),
        hist_attempted_unique=int(len(attempted_unique)),
        hist_solved_median_diff=float(hist_solved_median),
        hist_attempted_median_diff=float(hist_attempted_median),
        reco_k=int(k),
        reco_median_diff=float(reco_median_diff),
        reco_easy_share=float(reco_easy_share),
        reco_in_band_share=float(reco_in_band_share),
        reco_score_std=float(reco_score_std),
        flags=flags,
    )
    return diag, report, fig_paths


def _scan_users(
        reco: Recommender,
        *,
        user_ids: list[int],
        cutoff_pct: float,
        k: int,
        min_p: float,
        max_p: float,
        max_users: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for n, uid in enumerate(user_ids[: max(0, int(max_users))], start=1):
        try:
            diag, _, _ = _diagnose_one(
                reco,
                user_id=int(uid),
                cutoff_pct=float(cutoff_pct),
                k=int(k),
                min_p=float(min_p),
                max_p=float(max_p),
                plot=False,
                out_dir=REPORTS_DIR,
            )
        except Exception as e:
            rows.append(
                {
                    "user_id": int(uid),
                    "ok": 0,
                    "error": str(e),
                    "flags": "",
                }
            )
            continue
        rows.append(
            {
                "user_id": diag.user_id,
                "ok": 1,
                "cutoff_pct": diag.cutoff_pct,
                "cutoff_submission_id": diag.cutoff_submission_id,
                "hist_submissions": diag.hist_submissions,
                "hist_attempted_unique": diag.hist_attempted_unique,
                "hist_solved": diag.hist_solved,
                "hist_attempted_median_diff": diag.hist_attempted_median_diff,
                "hist_solved_median_diff": diag.hist_solved_median_diff,
                "reco_k": diag.reco_k,
                "reco_median_diff": diag.reco_median_diff,
                "reco_easy_share": diag.reco_easy_share,
                "reco_in_band_share": diag.reco_in_band_share,
                "reco_score_std": diag.reco_score_std,
                "flags": ",".join(diag.flags),
            }
        )
        if n % 50 == 0:
            print(f"[scan] {n}/{min(len(user_ids), int(max_users))} users done...")
    return pd.DataFrame(rows)


def main() -> int:
    """CLI 入口：单用户诊断或批量 scan，并输出到 Reports/diag 与 Reports/fig。"""
    ap = argparse.ArgumentParser(description="诊断“推荐偏简单/全是难度1”的原因，并提供可失败的校验模式。")
    ap.add_argument("--user-id", type=int, default=0, help="单个 user_id（与 --users/--scan 互斥）")
    ap.add_argument("--users", type=str, default="", help="逗号分隔 user_id 列表，例如：104,208,999")
    ap.add_argument("--scan", action="store_true", help="批量扫描（默认扫描所有出现过 submissions 的 user_id）")
    ap.add_argument("--max-users", type=int, default=300, help="批量扫描上限（避免跑太久）")
    ap.add_argument("--cutoff-pct", type=float, default=0.50, help="历史切片比例（0~1）")
    ap.add_argument("--min-p", type=float, default=0.40, help="成长带下限 min_p")
    ap.add_argument("--max-p", type=float, default=0.70, help="成长带上限 max_p")
    ap.add_argument("--k", type=int, default=10, help="Top-K")
    ap.add_argument("--plot", action="store_true", help="单用户模式输出图表到 Reports/fig/")
    ap.add_argument("--out", type=str, default="", help="输出路径（单用户=txt；scan=csv；默认写入 Reports/diag/）")
    ap.add_argument(
        "--fail-on",
        type=str,
        default="",
        help="逗号分隔：命中这些 flags 则退出码=2，例如：easy_bias_vs_history,score_plateau_topk",
    )
    ap.add_argument("--fail-if-any", action="store_true", help="scan 模式：任一用户 flags 非空则退出码=2")
    args = ap.parse_args()

    reco = Recommender()
    out_dir = REPORTS_DIAG_DIR

    fail_set = set([x.strip() for x in (args.fail_on or "").split(",") if x.strip()])

    user_ids = []
    if args.user_id:
        user_ids = [int(args.user_id)]
    user_ids.extend(_parse_user_list(args.users))
    user_ids = [int(x) for x in user_ids if int(x) > 0]
    user_ids = sorted(set(user_ids))

    if args.scan:
        if not user_ids:
            user_ids = sorted([int(x) for x in reco._subs_by_user.keys() if int(x) > 0])
        df = _scan_users(
            reco,
            user_ids=user_ids,
            cutoff_pct=float(args.cutoff_pct),
            k=int(args.k),
            min_p=float(args.min_p),
            max_p=float(args.max_p),
            max_users=int(args.max_users),
        )
        out_path = Path(args.out) if args.out else (out_dir / "diag_scan_users.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Wrote", out_path, "rows=", len(df))

        if args.fail_if_any:
            flags = df.get("flags", "")
            flags = flags.fillna("") if hasattr(flags, "fillna") else flags
            flagged = df[(df.get("ok", 0) == 1) & (flags.astype(str).str.strip() != "")]
            if len(flagged):
                print("FAIL: found flagged users =", int(len(flagged)))
                return 2
        return 0

    if len(user_ids) != 1:
        raise SystemExit("请指定单个 --user-id 或 --users（仅一个），或使用 --scan")

    user_id = int(user_ids[0])
    diag, report, figs = _diagnose_one(
        reco,
        user_id=user_id,
        cutoff_pct=float(args.cutoff_pct),
        k=int(args.k),
        min_p=float(args.min_p),
        max_p=float(args.max_p),
        plot=bool(args.plot),
        out_dir=out_dir,
    )

    out_path = Path(args.out) if args.out else (out_dir / f"diag_user_{int(user_id)}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print("Wrote", out_path)
    for p in figs:
        print("Wrote", p)

    if fail_set and any(f in fail_set for f in diag.flags):
        print("FAIL: matched flags =", ",".join([f for f in diag.flags if f in fail_set]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
