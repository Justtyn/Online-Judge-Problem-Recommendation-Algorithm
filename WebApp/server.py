import argparse
import base64
import html
import io
import json
import math
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "Reports"
CLEANDATA_DIR = ROOT / "CleanData"
FEATUREDATA_DIR = ROOT / "FeatureData"
MODELS_DIR = ROOT / "Models"
PIPELINE_PATH = MODELS_DIR / "pipeline_logreg.joblib"
RANDOM_SEED = 42


def parse_json_dict(s: str) -> dict[str, float]:
    s = (s or "").strip()
    if not s:
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


def normalize_dist(keys: list[str], d: dict[str, float]) -> dict[str, float]:
    if not keys:
        return {}
    vals = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
    s = sum(vals.values())
    if s <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / s for k, v in vals.items()}


class Recommender:
    def __init__(self) -> None:
        if not PIPELINE_PATH.exists():
            raise RuntimeError(
                f"找不到离线模型文件：{PIPELINE_PATH}\n"
                f"请先运行：\n"
                f"  python 04_build_features.py\n"
                f"  python 05_train_eval.py\n"
                f"以生成并保存 `Models/pipeline_logreg.joblib`。"
            )

        artifact = joblib.load(PIPELINE_PATH)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.model = artifact["pipeline"]
            self.feature_cols = list(artifact.get("feature_cols") or [])
        else:
            self.model = artifact
            self.feature_cols = []

        if not self.feature_cols:
            train_samples = pd.read_csv(FEATUREDATA_DIR / "train_samples.csv")
            self.feature_cols = [
                c
                for c in train_samples.columns
                if c not in {"ac", "submission_id", "user_id", "problem_id"}
            ]

        self.col_to_idx = {c: i for i, c in enumerate(self.feature_cols)}

        required = {"attempt_no", "difficulty_filled", "level", "perseverance", "lang_match", "tag_match"}
        missing = sorted(required - set(self.feature_cols))
        if missing:
            raise RuntimeError(
                f"离线模型/特征不匹配：缺少特征列 {missing}；请重新运行 `python 04_build_features.py` 与 `python 05_train_eval.py`。"
            )

        self.lang_cols = [c for c in self.feature_cols if c.startswith("lang_")]
        self.tag_cols = [c for c in self.feature_cols if c.startswith("tag_")]
        self.lang_names = [c.removeprefix("lang_") for c in self.lang_cols]
        self.tag_names = [c.removeprefix("tag_") for c in self.tag_cols]
        self.lang_idx = [self.col_to_idx[c] for c in self.lang_cols]
        self.tag_idx = [self.col_to_idx[c] for c in self.tag_cols]

        self.numeric_idx = {
            "attempt_no": self.col_to_idx["attempt_no"],
            "difficulty_filled": self.col_to_idx["difficulty_filled"],
            "level": self.col_to_idx["level"],
            "perseverance": self.col_to_idx["perseverance"],
            "lang_match": self.col_to_idx["lang_match"],
            "tag_match": self.col_to_idx["tag_match"],
        }

        self.problems = pd.read_csv(CLEANDATA_DIR / "problems.csv")
        self.problems["problem_id"] = pd.to_numeric(self.problems["problem_id"], errors="coerce").astype(int)
        if "title" not in self.problems.columns:
            self.problems["title"] = ""
        self.problems["difficulty"] = pd.to_numeric(self.problems["difficulty"], errors="coerce")
        diff_median = int(np.nanmedian(self.problems["difficulty"])) if self.problems["difficulty"].notna().any() else 5
        self.problems["difficulty_filled"] = self.problems["difficulty"].fillna(diff_median).astype(int)

        self.tag_to_j = {t: j for j, t in enumerate(self.tag_names)}

        def parse_tags_cell(x) -> list[str]:
            s = str(x or "").strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    v = json.loads(s)
                    if isinstance(v, list):
                        return [str(t) for t in v]
                except Exception:
                    pass
            # fallback: comma-separated
            parts = [p.strip() for p in s.strip("[]").split(",") if p.strip()]
            return parts

        self.problems["tags_list"] = self.problems["tags"].apply(parse_tags_cell)
        self.problems["tags2"] = self.problems["tags_list"].apply(
            lambda x: [str(t) for t in (x[:2] if isinstance(x, list) else [])]
        )
        self.problem_ids = self.problems["problem_id"].to_numpy(dtype=np.int32)
        self.problem_diff = self.problems["difficulty_filled"].to_numpy(dtype=np.int32)
        self.problem_title = self.problems["title"].astype(str).fillna("").to_numpy(dtype=object)
        self.problem_tags2 = self.problems["tags2"].to_list()
        self.pid_to_i = {int(pid): i for i, pid in enumerate(self.problem_ids.tolist())}
        self.problem_tags_mh = np.zeros((len(self.problems), len(self.tag_names)), dtype=np.uint8)
        for i, tags_list in enumerate(self.problems["tags_list"].tolist()):
            if not isinstance(tags_list, list):
                continue
            for t in tags_list[:2]:
                j = self.tag_to_j.get(str(t))
                if j is not None:
                    self.problem_tags_mh[i, j] = 1
        self.problem_tag_counts = self.problem_tags_mh.sum(axis=1).astype(np.float32)

        self._subs = pd.read_csv(CLEANDATA_DIR / "submissions.csv", low_memory=False)
        for col in ("submission_id", "user_id", "problem_id", "attempt_no", "ac"):
            if col in self._subs.columns:
                self._subs[col] = pd.to_numeric(self._subs[col], errors="coerce")
        self._subs["submission_id"] = self._subs["submission_id"].fillna(0).astype(int)
        self._subs["user_id"] = self._subs["user_id"].fillna(0).astype(int)
        self._subs["problem_id"] = self._subs["problem_id"].fillna(0).astype(int)
        self._subs["attempt_no"] = self._subs["attempt_no"].fillna(1).astype(int)
        self._subs["ac"] = self._subs["ac"].fillna(0).astype(int)
        if "language" not in self._subs.columns:
            self._subs["language"] = ""
        self._subs["language"] = self._subs["language"].astype(str).fillna("")

        self._subs = self._subs.sort_values(["user_id", "submission_id"]).reset_index(drop=True)
        self._subs_by_user: dict[int, pd.DataFrame] = {
            int(uid): g[["submission_id", "problem_id", "language", "attempt_no", "ac"]].copy()
            for uid, g in self._subs.groupby("user_id", sort=False)
            if int(uid) > 0
        }

        # perseverance 归一化用固定分母（全量用户的 avg_attempts_per_problem 的 P95）
        up = self._subs.groupby(["user_id", "problem_id"], as_index=False).agg(n_attempts=("submission_id", "count"))
        avg_attempts = up.groupby("user_id")["n_attempts"].mean()
        p95 = float(np.percentile(avg_attempts.values, 95)) if len(avg_attempts) else 1.0
        self._perseverance_denom = math.log1p(p95) if p95 > 0 else 1.0

    @staticmethod
    def _fig_to_b64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _rank_by_growth_band(
            cand_idx: np.ndarray,
            *,
            score: np.ndarray,
            difficulty: np.ndarray,
            tag_match: np.ndarray | None,
            in_band: np.ndarray,
            min_p: float,
            max_p: float,
            target_alpha: float = 0.20,
    ) -> np.ndarray:
        if len(cand_idx) == 0:
            return cand_idx

        min_p = float(min_p)
        max_p = float(max_p)
        if max_p < min_p:
            min_p, max_p = max_p, min_p

        target_alpha = float(max(0.0, min(1.0, target_alpha)))
        target_p = min_p + target_alpha * (max_p - min_p)

        score = np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        difficulty = difficulty.astype(np.int32, copy=False)
        in_band = in_band.astype(bool, copy=False)

        dist_to_band = np.where(
            in_band,
            0.0,
            np.minimum(np.abs(score - min_p), np.abs(score - max_p)),
        ).astype(np.float32)
        dist_to_target = np.abs(score - float(target_p)).astype(np.float32)

        if tag_match is None:
            tag_key = np.zeros((len(cand_idx),), dtype=np.float32)
        else:
            tag_key = -np.nan_to_num(tag_match.astype(np.float32), nan=0.0)[cand_idx]

        order = np.lexsort(
            (
                # tie-breaker (stable/deterministic)
                np.asarray(cand_idx, dtype=np.int32),
                # 更有挑战：同条件下优先更低的 P(AC)
                score[cand_idx],
                # 更贴合偏好：tag_match 越大越优先
                tag_key,
                # 更有挑战：难度越高越优先
                -difficulty[cand_idx],
                # 成长带内：优先靠近目标成功率（默认偏向下限）
                dist_to_target[cand_idx],
                # 带外补齐：优先离成长带最近
                dist_to_band[cand_idx],
                # 成长带优先（0=带内,1=带外）
                (~in_band[cand_idx]).astype(np.int8),
            )
        )
        return cand_idx[order]

    def _user_cutoff_id(self, user_id: int, pct: float) -> int:
        user_df = self._subs_by_user.get(int(user_id))
        if user_df is None or user_df.empty:
            raise RuntimeError(f"user_id={user_id} 没有 submissions 记录")
        pct = float(max(0.0, min(1.0, pct)))
        ids = user_df["submission_id"].to_numpy(dtype=np.int64)
        idx = int(round(pct * (len(ids) - 1))) if len(ids) > 1 else 0
        return int(ids[idx])

    def _profile_from_history(
            self, user_df: pd.DataFrame, cutoff_id: int
    ) -> tuple[float, float, dict[str, float], dict[str, float], set[int], dict[int, int]]:
        hist = user_df[user_df["submission_id"] <= int(cutoff_id)]
        if hist.empty:
            return 0.0, 0.0, {}, {}, set(), {}

        solved = set(hist.loc[hist["ac"] == 1, "problem_id"].astype(int).tolist())
        attempts = hist.groupby("problem_id").size().astype(int)
        attempt_next_map = {int(pid): int(n) + 1 for pid, n in attempts.items()}

        up = hist.groupby("problem_id", as_index=False).agg(
            n_attempts=("submission_id", "count"),
            solved=("ac", "max"),
        )
        up["difficulty_filled"] = up["problem_id"].map(
            lambda pid: int(self.problem_diff[self.pid_to_i[int(pid)]]) if int(pid) in self.pid_to_i else 5
        )
        up["diff_norm"] = up["difficulty_filled"].astype(float) / 10.0
        num = float((up["solved"].astype(float) * up["diff_norm"]).sum())
        den = float(up["diff_norm"].sum())
        level = float(num / (den + 1e-9))
        level = float(max(0.0, min(1.0, level)))

        avg_attempts = float(up["n_attempts"].mean()) if len(up) else 0.0
        perseverance = math.log1p(avg_attempts) / self._perseverance_denom if self._perseverance_denom > 0 else 0.0
        perseverance = float(max(0.0, min(1.0, perseverance)))

        # lang pref
        lang_counts = hist.groupby("language").size().to_dict()
        lang_pref = {l: float(lang_counts.get(l, 0.0)) for l in self.lang_names}
        lang_pref = normalize_dist(self.lang_names, lang_pref)

        # tag pref（按做过的题的 tags2 计数）
        tag_counts: dict[str, float] = {t: 0.0 for t in self.tag_names}
        for pid in up["problem_id"].astype(int).tolist():
            i = self.pid_to_i.get(int(pid))
            if i is None:
                continue
            for t in self.problem_tags2[i]:
                if t in tag_counts:
                    tag_counts[t] += 1.0
        tag_pref = normalize_dist(self.tag_names, tag_counts)
        return level, perseverance, lang_pref, tag_pref, solved, attempt_next_map

    def recommend_for_user_history(
            self,
            *,
            user_id: int,
            cutoff_pct: float,
            k: int,
            min_p: float,
            max_p: float,
    ) -> tuple[dict, list[dict]]:
        user_id = int(user_id)
        user_df = self._subs_by_user.get(user_id)
        if user_df is None or user_df.empty:
            raise RuntimeError(f"user_id={user_id} 不存在或无 submissions")

        cutoff_id = self._user_cutoff_id(user_id, cutoff_pct)
        level, perseverance, lang_pref, tag_pref, solved, attempt_next_map = self._profile_from_history(user_df,
                                                                                                        cutoff_id)

        k = int(max(1, min(50, k)))
        min_p = float(max(0.0, min(1.0, min_p)))
        max_p = float(max(0.0, min(1.0, max_p)))
        if max_p < min_p:
            min_p, max_p = max_p, min_p

        chosen_lang = ""
        if self.lang_names:
            chosen_lang = max(self.lang_names, key=lambda l: float(lang_pref.get(l, 0.0)))

        attempt_no_vec = np.ones((len(self.problem_ids),), dtype=np.float32)
        for pid, next_no in attempt_next_map.items():
            i = self.pid_to_i.get(int(pid))
            if i is not None:
                attempt_no_vec[i] = float(max(1, min(10, int(next_no))))

        solved_mask = (
            np.isin(self.problem_ids, np.fromiter((int(x) for x in solved), dtype=np.int32))
            if solved
            else np.zeros((len(self.problem_ids),), dtype=bool)
        )
        candidate_mask = ~solved_mask

        tag_pref_vec = np.asarray([float(tag_pref.get(t, 0.0)) for t in self.tag_names], dtype=np.float32)
        tm_sum = (self.problem_tags_mh.astype(np.float32) * tag_pref_vec).sum(axis=1)
        tm_den = np.maximum(1.0, self.problem_tag_counts)
        tag_match = (tm_sum / tm_den).astype(np.float32)

        X = np.zeros((len(self.problem_ids), len(self.feature_cols)), dtype=np.float32)
        X[:, self.numeric_idx["attempt_no"]] = attempt_no_vec
        X[:, self.numeric_idx["difficulty_filled"]] = self.problem_diff.astype(np.float32)
        X[:, self.numeric_idx["level"]] = float(level)
        X[:, self.numeric_idx["perseverance"]] = float(perseverance)
        X[:, self.numeric_idx["tag_match"]] = tag_match
        X[:, self.tag_idx] = self.problem_tags_mh.astype(np.float32)

        lang_match_val = float(lang_pref.get(chosen_lang, 0.0)) if chosen_lang else 0.0
        X[:, self.numeric_idx["lang_match"]] = lang_match_val
        if self.lang_idx and chosen_lang in self.lang_names:
            X[:, self.lang_idx] = 0.0
            X[:, self.lang_idx[self.lang_names.index(chosen_lang)]] = 1.0

        score = self.model.predict_proba(X)[:, 1].astype(np.float32)
        in_band = (score >= float(min_p)) & (score <= float(max_p))

        idx_all = np.where(candidate_mask)[0]
        if len(idx_all) == 0:
            raise RuntimeError(f"user_id={user_id} 已 AC 全部题目或无候选集")

        ranked = self._rank_by_growth_band(
            idx_all,
            score=score,
            difficulty=self.problem_diff,
            tag_match=tag_match,
            in_band=in_band,
            min_p=min_p,
            max_p=max_p,
        )
        picks = ranked[:k]

        rec_rows: list[dict] = []
        for rank, i in enumerate(picks.tolist(), start=1):
            rec_rows.append(
                {
                    "rank": int(rank),
                    "problem_id": int(self.problem_ids[i]),
                    "title": str(self.problem_title[i] or ""),
                    "difficulty": int(self.problem_diff[i]),
                    "tags": ",".join(self.problem_tags2[i]) if i < len(self.problem_tags2) else "",
                    "p_ac": float(score[i]),
                    "language": chosen_lang,
                    "in_growth_band": int(bool(in_band[i])),
                }
            )

        meta = {
            "user_id": int(user_id),
            "cutoff_pct": float(cutoff_pct),
            "cutoff_submission_id": int(cutoff_id),
            "hist_submissions": int((user_df["submission_id"] <= int(cutoff_id)).sum()),
            "hist_solved": int(len(solved)),
            "level": float(level),
            "perseverance": float(perseverance),
            "top_language": str(chosen_lang),
            "zpd": [float(min_p), float(max_p)],
        }
        return meta, rec_rows

    def student_dashboard_payload(
            self,
            *,
            user_id: int,
            cutoff_pct: float,
            k: int,
            min_p: float,
            max_p: float,
    ) -> dict:
        user_id = int(user_id)
        user_df = self._subs_by_user.get(user_id)
        if user_df is None or user_df.empty:
            raise RuntimeError(f"user_id={user_id} 不存在或无 submissions")

        meta, rec_rows = self.recommend_for_user_history(
            user_id=user_id,
            cutoff_pct=cutoff_pct,
            k=k,
            min_p=min_p,
            max_p=max_p,
        )
        cutoff_id = int(meta["cutoff_submission_id"])

        hist = user_df[user_df["submission_id"] <= cutoff_id].copy()
        if not hist.empty:
            hist["difficulty"] = hist["problem_id"].map(
                lambda pid: int(self.problem_diff[self.pid_to_i[int(pid)]]) if int(pid) in self.pid_to_i else 5
            )
        else:
            hist["difficulty"] = []

        # 1) 时间轴散点：历史提交（难度 vs 时间）+ cutoff + 推荐题难度点
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.set_title(f"时间轴散点：user_id={user_id}（历史提交难度 & 推荐题难度）")
        if not hist.empty:
            ok = hist["ac"].astype(int).to_numpy() == 1
            x = hist["submission_id"].to_numpy(dtype=np.int64)
            y = hist["difficulty"].to_numpy(dtype=np.int32)
            ax.scatter(x[~ok], y[~ok], s=14, alpha=0.35, c="#ef4444", label="未AC")
            ax.scatter(x[ok], y[ok], s=14, alpha=0.35, c="#22c55e", label="AC")
        ax.axvline(cutoff_id, color="#2563eb", lw=2, alpha=0.9, label="cutoff")
        reco_diffs = [int(r["difficulty"]) for r in rec_rows]
        if reco_diffs:
            ax.scatter(
                np.full((len(reco_diffs),), cutoff_id, dtype=np.int64),
                np.asarray(reco_diffs, dtype=np.int32),
                marker="*",
                s=140,
                c="#1d4ed8",
                edgecolors="white",
                linewidths=0.8,
                label="推荐Top-K",
                zorder=5,
            )
        ax.set_xlabel("submission_id（时间近似）")
        ax.set_ylabel("题目难度（1-10）")
        ax.set_yticks(range(1, 11))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", ncols=4, frameon=False)
        timeline_b64 = self._fig_to_b64(fig)

        # 2) 雷达对比：语言&标签（历史 vs 推荐）
        hist_lang_counts = hist.groupby("language").size().to_dict() if not hist.empty else {}
        hist_lang = np.asarray([float(hist_lang_counts.get(l, 0.0)) for l in self.lang_names], dtype=np.float32)
        hist_lang = hist_lang / (hist_lang.sum() if hist_lang.sum() > 0 else 1.0)

        hist_tag_counts = {t: 0.0 for t in self.tag_names}
        if not hist.empty:
            for pid in hist["problem_id"].astype(int).unique().tolist():
                i = self.pid_to_i.get(int(pid))
                if i is None:
                    continue
                for t in self.problem_tags2[i]:
                    if t in hist_tag_counts:
                        hist_tag_counts[t] += 1.0
        hist_tag = np.asarray([float(hist_tag_counts.get(t, 0.0)) for t in self.tag_names], dtype=np.float32)
        hist_tag = hist_tag / (hist_tag.sum() if hist_tag.sum() > 0 else 1.0)

        reco_lang = np.zeros((len(self.lang_names),), dtype=np.float32)
        if self.lang_names and meta.get("top_language") in self.lang_names:
            reco_lang[self.lang_names.index(str(meta["top_language"]))] = 1.0

        reco_tag_counts = {t: 0.0 for t in self.tag_names}
        for r in rec_rows:
            pid = int(r["problem_id"])
            i = self.pid_to_i.get(pid)
            if i is None:
                continue
            for t in self.problem_tags2[i]:
                if t in reco_tag_counts:
                    reco_tag_counts[t] += 1.0
        reco_tag = np.asarray([float(reco_tag_counts.get(t, 0.0)) for t in self.tag_names], dtype=np.float32)
        reco_tag = reco_tag / (reco_tag.sum() if reco_tag.sum() > 0 else 1.0)

        def radar(ax, labels: list[str], v1: np.ndarray, v2: np.ndarray, title: str) -> None:
            n = len(labels)
            if n == 0:
                ax.set_axis_off()
                return
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            angles = np.concatenate([angles, angles[:1]])
            v1c = np.concatenate([v1, v1[:1]])
            v2c = np.concatenate([v2, v2[:1]])
            ax.plot(angles, v1c, color="#2563eb", lw=2, label="历史")
            ax.fill(angles, v1c, color="#2563eb", alpha=0.10)
            ax.plot(angles, v2c, color="#f59e0b", lw=2, label="推荐Top-K")
            ax.fill(angles, v2c, color="#f59e0b", alpha=0.10)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
            ax.set_title(title, pad=14)
            ax.grid(True, alpha=0.25)

        fig = plt.figure(figsize=(10, 4.6))
        ax1 = fig.add_subplot(1, 2, 1, projection="polar")
        radar(ax1, self.lang_names, hist_lang, reco_lang, "雷达对比：语言分布")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        radar(ax2, self.tag_names, hist_tag, reco_tag, "雷达对比：标签分布")
        ax1.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.25), frameon=False, ncols=2)
        radar_b64 = self._fig_to_b64(fig)

        # 3) 难度阶梯：按推荐 rank 展示 difficulty & P(AC)
        ranks = np.asarray([int(r["rank"]) for r in rec_rows], dtype=np.int32)
        diffs = np.asarray([int(r["difficulty"]) for r in rec_rows], dtype=np.int32)
        ps = np.asarray([float(r["p_ac"]) for r in rec_rows], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.set_title("难度阶梯：推荐列表（rank→difficulty，并用颜色表示 P(AC)）")
        ax.plot(ranks, diffs, color="#334155", lw=1.5, alpha=0.7)
        sc = ax.scatter(ranks, diffs, c=ps, cmap="viridis", s=80, edgecolors="white", linewidths=0.6)
        ax.set_xlabel("推荐 rank（1=最高分）")
        ax.set_ylabel("题目难度（1-10）")
        ax.set_yticks(range(1, 11))
        ax.set_xticks(ranks.tolist())
        ax.grid(True, alpha=0.25, linestyle="--")
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("P(AC)")
        ladder_b64 = self._fig_to_b64(fig)

        return {
            "meta": meta,
            "images": {
                "timeline_scatter": timeline_b64,
                "radar_compare": radar_b64,
                "difficulty_ladder": ladder_b64,
            },
            "recommendations": rec_rows,
        }

    def recommend(
            self,
            *,
            level: float,
            perseverance: float,
            lang_pref: dict[str, float],
            tag_pref: dict[str, float],
            k: int,
            attempt_no: int,
            min_p: float,
            max_p: float,
            mode: str,
    ) -> tuple[pd.DataFrame, str]:
        level = float(max(0.0, min(1.0, level)))
        perseverance = float(max(0.0, min(1.0, perseverance)))
        k = int(max(1, min(50, k)))
        attempt_no = int(max(1, min(10, attempt_no)))
        min_p = float(max(0.0, min(1.0, min_p)))
        max_p = float(max(0.0, min(1.0, max_p)))
        if max_p < min_p:
            min_p, max_p = max_p, min_p

        lang_pref = normalize_dist(self.lang_names, lang_pref)
        tag_pref = normalize_dist(self.tag_names, tag_pref)
        tag_pref_vec = np.asarray([float(tag_pref.get(t, 0.0)) for t in self.tag_names], dtype=np.float32)

        base = np.zeros((len(self.problems), len(self.feature_cols)), dtype=np.float32)
        base[:, self.numeric_idx["attempt_no"]] = float(attempt_no)
        base[:, self.numeric_idx["difficulty_filled"]] = self.problem_diff.astype(np.float32)
        base[:, self.numeric_idx["level"]] = float(level)
        base[:, self.numeric_idx["perseverance"]] = float(perseverance)
        base[:, self.tag_idx] = self.problem_tags_mh.astype(np.float32)

        tm_sum = (self.problem_tags_mh.astype(np.float32) * tag_pref_vec).sum(axis=1)
        tm_den = np.maximum(1.0, self.problem_tag_counts)
        base[:, self.numeric_idx["tag_match"]] = tm_sum / tm_den

        probs_by_lang: dict[str, np.ndarray] = {}
        for lang in self.lang_names:
            X = base.copy()
            X[:, self.numeric_idx["lang_match"]] = float(lang_pref.get(lang, 0.0))
            if self.lang_idx:
                X[:, self.lang_idx] = 0.0
                if lang in self.lang_names:
                    X[:, self.lang_idx[self.lang_names.index(lang)]] = 1.0
            probs_by_lang[lang] = self.model.predict_proba(X)[:, 1].astype(np.float32)

        if mode == "best":
            best_lang = np.array(self.lang_names, dtype=object)[
                np.argmax(np.vstack([probs_by_lang[l] for l in self.lang_names]), axis=0)
            ]
            score = np.max(np.vstack([probs_by_lang[l] for l in self.lang_names]), axis=0)
            chosen_lang = best_lang
        else:
            score = np.zeros((len(self.problems),), dtype=np.float32)
            for lang in self.lang_names:
                score += float(lang_pref.get(lang, 0.0)) * probs_by_lang[lang]
            chosen_lang = np.array(
                [max(self.lang_names, key=lambda l: probs_by_lang[l][i]) for i in range(len(score))],
                dtype=object,
            )

        df = self.problems[["problem_id", "title", "difficulty_filled"]].copy()
        df["tags"] = self.problems["tags_list"].apply(lambda x: ",".join(x[:2]) if isinstance(x, list) else "")
        df["score"] = score
        df["language"] = chosen_lang
        in_band = (df["score"].to_numpy(dtype=np.float32) >= float(min_p)) & (
            df["score"].to_numpy(dtype=np.float32) <= float(max_p)
        )
        tag_match_vec = base[:, self.numeric_idx["tag_match"]].astype(np.float32)
        ranked = self._rank_by_growth_band(
            np.arange(len(df), dtype=np.int32),
            score=df["score"].to_numpy(dtype=np.float32),
            difficulty=self.problem_diff,
            tag_match=tag_match_vec,
            in_band=in_band,
            min_p=min_p,
            max_p=max_p,
        )
        out = df.iloc[ranked[:k]].copy()

        fig = plt.figure(figsize=(8, 5))
        plt.style.use('ggplot')  # 使用更好看的绘图风格
        plt.hist(out["difficulty_filled"].astype(int), bins=np.arange(0.5, 10.6, 1),
                 edgecolor="white", alpha=0.8, color="#3b82f6")
        plt.title("推荐题目难度分布", fontsize=14)
        plt.xlabel("难度（1-10）", fontsize=12)
        plt.ylabel("数量", fontsize=12)
        plt.xticks(range(1, 11))
        plt.grid(True, linestyle='--', alpha=0.5)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=160, transparent=False)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return out, img_b64


RECO: Recommender | None = None

# ----------------------------------------------------------------------------
# 现代化 CSS 样式 (支持深色模式)
# ----------------------------------------------------------------------------
STYLE_CSS = """
:root {
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --primary-light: rgba(59, 130, 246, 0.1);

    --bg-body: #f8fafc;
    --bg-card: #ffffff;
    --bg-input: #ffffff;

    --text-main: #0f172a;
    --text-muted: #64748b;
    --text-inverse: #ffffff;

    --border: #e2e8f0;
    --border-hover: #cbd5e1;

    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --radius: 12px;

    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-body: #0f172a;
        --bg-card: #1e293b;
        --bg-input: #334155;

        --text-main: #f8fafc;
        --text-muted: #94a3b8;

        --border: #334155;
        --border-hover: #475569;

        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3);
    }
}

/* 强制覆盖深色模式的类 (JS toggle) */
html.dark-mode {
    --bg-body: #0f172a;
    --bg-card: #1e293b;
    --bg-input: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --border-hover: #475569;
}
html.light-mode {
    --bg-body: #f8fafc;
    --bg-card: #ffffff;
    --bg-input: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --border-hover: #cbd5e1;
}

body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--bg-body);
    color: var(--text-main);
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    transition: background-color 0.3s, color 0.3s;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 16px;
}

/* Header */
header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}

h1 { font-size: 1.8rem; font-weight: 700; margin: 0; letter-spacing: -0.025em; }
h2 { font-size: 1.4rem; font-weight: 600; margin: 1.5rem 0 1rem; }
h3 { font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem; }

a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
a:hover { color: var(--primary-hover); }

/* Components */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s, box-shadow 0.2s, background-color 0.3s, border-color 0.3s;
}

.card:hover {
    box-shadow: var(--shadow);
    transform: translateY(-2px);
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem;
}

.subgrid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.5rem;
}

/* Non-home pages: full-width blocks stacked vertically */
body.page-student .grid,
body.page-custom .grid {
    grid-template-columns: 1fr;
}
body.page-student .subgrid,
body.page-custom .subgrid {
    grid-template-columns: 1fr;
}

/* Form Elements */
label {
    display: block;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: var(--text-muted);
}

input[type="text"], input[type="number"], select, textarea {
    width: 100%;
    padding: 0.6rem 0.8rem;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-main);
    font-family: inherit;
    font-size: 0.95rem;
    box-sizing: border-box;
    transition: border-color 0.2s, box-shadow 0.2s;
}

input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-light);
}

/* Range Slider */
input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    background: transparent;
    margin: 10px 0;
}
input[type=range]:focus { outline: none; }

/* Webkit Slider */
input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 6px;
    cursor: pointer;
    background: var(--border);
    border-radius: 99px;
    transition: background 0.2s;
}
input[type=range]::-webkit-slider-thumb {
    height: 18px;
    width: 18px;
    border-radius: 50%;
    background: var(--primary);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    border: 2px solid var(--bg-card);
    transition: transform 0.1s;
}
input[type=range]:focus::-webkit-slider-thumb {
    transform: scale(1.1);
    box-shadow: 0 0 0 3px var(--primary-light);
}

/* Firefox Slider */
input[type=range]::-moz-range-track {
    width: 100%;
    height: 6px;
    cursor: pointer;
    background: var(--border);
    border-radius: 99px;
}
input[type=range]::-moz-range-thumb {
    height: 18px;
    width: 18px;
    border: 2px solid var(--bg-card);
    border-radius: 50%;
    background: var(--primary);
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* Buttons */
.actions {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
}

button, .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
    text-decoration: none;
}

button[type="submit"], .btn-primary {
    background-color: var(--primary);
    color: white;
}
button[type="submit"]:hover, .btn-primary:hover {
    background-color: var(--primary-hover);
    box-shadow: 0 4px 12px var(--primary-light);
}

.btn-secondary {
    background-color: var(--bg-card);
    color: var(--text-main);
    border-color: var(--border);
}
.btn-secondary:hover {
    background-color: var(--bg-body);
    border-color: var(--border-hover);
}

/* Tables */
.table-wrapper {
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    white-space: nowrap;
}
th {
    background-color: var(--bg-body);
    font-weight: 600;
    text-align: left;
    padding: 12px 16px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
}
td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-main);
}
tr:last-child td { border-bottom: none; }
tr:hover td { background-color: var(--primary-light); }

/* Visuals */
.chart-container {
    background: white; /* 保持图表底色为白，确保Matplotlib渲染清晰 */
    padding: 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    display: flex;
    justify-content: center;
    align-items: center;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
}

/* Utils */
.muted { color: var(--text-muted); font-size: 0.85rem; }
.help { margin-top: 6px; color: var(--text-muted); font-size: 0.85rem; }
.viz-img { width: 100%; height: 360px; object-fit: contain; }
.viz-img-lg { width: 100%; height: 320px; object-fit: contain; }
.multiselect { min-height: 210px; }
.row { display: flex; align-items: center; gap: 10px; }
.row output { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--primary); width: 45px; text-align: right; }
.pill {
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.75rem;
    border-radius: 99px;
    background: var(--bg-body);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
}
.theme-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: var(--text-muted);
    padding: 0;
}
.theme-toggle:hover {
    background-color: var(--bg-body);
    color: var(--primary);
}

details {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
    background: var(--bg-body);
}
details summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--text-main);
    user-select: none;
}
details[open] { background: var(--bg-card); }

/* Home: dashboard */
.toolbar {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1rem;
    align-items: end;
}
.toolbar .left {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 1rem;
}
.toolbar .right {
    display: flex;
    gap: 0.6rem;
    justify-content: flex-end;
    flex-wrap: wrap;
}
.toolbar-simple {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
}
.toolbar-simple .topline {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.toolbar-simple .filters {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.8rem;
}
.fig-summary {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 2px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-body);
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}
.chip.primary { border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.12); color: var(--primary-hover); }
.chip.success { border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); color: var(--success); }
.chip.warn { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.12); color: var(--warning); }
.chip.danger { border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.12); color: var(--danger); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
}
.card-actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.btn-sm { padding: 0.45rem 0.75rem; font-size: 0.85rem; border-radius: 10px; }
.img-frame {
    background: #fff;
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 10px;
    margin: 10px 0;
}
.fig-card h3 { margin: 0.35rem 0 0.5rem; }
.fig-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.section-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 1.8rem;
}
.section-header h2 { margin: 0; }
.section-desc { margin: 0.35rem 0 0; color: var(--text-muted); font-size: 0.92rem; }
.sticky { position: sticky; top: 12px; z-index: 20; }
.hidden { display: none !important; }

/* Modal (home: click image -> large view + explanation) */
.modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 18px;
    z-index: 999;
}
.modal {
    width: min(1180px, 100%);
    max-height: min(92vh, 980px);
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.modal-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
}
.modal-header h3 { margin: 6px 0 2px; }
.modal-close {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    width: 38px;
    height: 38px;
    border-radius: 10px;
    font-size: 20px;
    line-height: 1;
}
.modal-close:hover { color: var(--text-main); border-color: var(--border-hover); }
.modal-body {
    display: grid;
    grid-template-columns: 1.6fr 1fr;
    gap: 0;
    min-height: 0;
}
.modal-left {
    background: #fff;
    border-right: 1px solid var(--border);
    padding: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
}
.modal-left img {
    width: 100%;
    height: 100%;
    max-height: 78vh;
    object-fit: contain;
}
.modal-right {
    padding: 14px 16px;
    overflow: auto;
}
.modal-right h4 { margin: 0 0 8px; font-size: 1rem; }
.modal-ul { margin: 8px 0 0; padding-left: 18px; color: var(--text-muted); font-size: 0.9rem; }
.modal-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.img-button {
    width: 100%;
    padding: 0;
    background: transparent;
    border: none;
    cursor: zoom-in;
}
@media (max-width: 920px) {
    .modal-body { grid-template-columns: 1fr; }
    .modal-left { border-right: none; border-bottom: 1px solid var(--border); }
}
"""

FIG_INFO: dict[str, dict[str, object]] = {
    # A. 数据层（训练前）
    "fig_level_hist.png": {
        "title": "level 分布（能力画像是否有区分度）",
        "name": "level_hist",
        "section": "A",
        "tags": ["画像", "合理性检查"],
        "summary": "用来检验能力画像 level 是否能把用户区分开（而不是全部挤在 0 或 1）。",
        "how": [
            "横轴是 level（0~1）：越靠右代表越强；纵轴是人数。",
            "如果图像非常“尖”（几乎都在同一段），说明画像区分度不足或归一化不合理。",
        ],
        "tips": [
            "低活跃用户占比高会让画像更噪（可结合 fig_user_activity.png 理解）。",
        ],
    },
    "fig_perseverance_hist.png": {
        "title": "perseverance 分布（坚持/重试画像）",
        "name": "perseverance_hist",
        "section": "A",
        "tags": ["画像", "合理性检查"],
        "summary": "用来观察用户坚持度（重试倾向）的差异，避免全部接近 0 或 1。",
        "how": [
            "横轴是 perseverance（0~1）：越大表示更愿意重试/更“耐心”。",
            "如果大量用户都接近 1，通常是归一化尺度设置过小导致饱和。",
        ],
    },
    "fig_lang_dist.png": {
        "title": "语言分布（按提交次数）",
        "name": "language_dist",
        "section": "A",
        "tags": ["数据分布", "特征有效性"],
        "summary": "检查语言总体占比是否符合常识，也用于说明语言特征有“可学习”的差异。",
        "how": [
            "柱越高表示该语言提交越多；极端偏斜可能影响模型（某些语言 one-hot 近似无样本）。",
        ],
    },
    "fig_tag_dist.png": {
        "title": "标签分布（题型占比）",
        "name": "tag_dist",
        "section": "A",
        "tags": ["数据分布", "特征有效性"],
        "summary": "检查题库题型是否极端失衡；过度失衡会让模型/推荐更同质化。",
        "how": [
            "柱越高表示该标签出现越多；如果少数标签压倒性占比，需要在报告里说明影响。",
        ],
    },
    "fig_user_activity.png": {
        "title": "用户活跃度分布（提交次数长尾）",
        "name": "user_activity",
        "section": "A",
        "tags": ["数据分布", "长尾"],
        "summary": "展示典型长尾：少数高活跃用户贡献大量提交，大量用户只有少量记录。",
        "how": [
            "横轴是提交次数，纵轴是人数；长尾越明显，冷启动/画像稳定性问题越突出。",
        ],
    },
    "fig_difficulty_vs_ac.png": {
        "title": "难度 vs AC 率（合理性校验）",
        "name": "difficulty_vs_ac",
        "section": "A",
        "tags": ["合理性检查", "难度口径"],
        "summary": "检验难度标注是否可信：通常难度越高，AC 率应整体下降。",
        "how": [
            "横轴是难度（1~10），纵轴是 AC 率（0~1）。",
            "如果不降反升或大幅抖动，可能是 difficulty 质量/样本量问题。",
        ],
    },
    "fig_attemptno_vs_ac.png": {
        "title": "尝试次数 vs AC 率（学习/难度效应）",
        "name": "attemptno_vs_ac",
        "section": "A",
        "tags": ["特征解释", "学习效应"],
        "summary": "解释 attempt_no 与成功率关系：可能存在“越试越会”，也可能是“难题才会多次尝试”。",
        "how": [
            "横轴是 attempt_no（第几次尝试），纵轴是 AC 率。",
            "趋势需要结合业务解释：上升=学习效应；下降=更难题带来更多尝试。",
        ],
    },
    "fig_tag_acrate.png": {
        "title": "不同标签的平均 AC 率（题型差异）",
        "name": "tag_acrate",
        "section": "A",
        "tags": ["特征有效性"],
        "summary": "展示不同题型的平均通过率差异，用于说明标签特征有信息量。",
        "how": [
            "若所有标签 AC 率几乎相同，说明标签区分信息较弱或统计口径有误。",
        ],
    },
    "fig_lang_acrate.png": {
        "title": "不同语言的平均 AC 率（相关性，不是因果）",
        "name": "lang_acrate",
        "section": "A",
        "tags": ["特征有效性"],
        "summary": "检查语言与通过率是否有关联（更多反映用户群体/题目选择偏差，不建议因果解读）。",
        "how": [
            "差异存在不代表“某语言更强”，而可能是强者更偏好某语言。",
        ],
    },
    # B. 训练层（训练后）
    "fig_model_f1_compare.png": {
        "title": "模型 F1 对比（时间切分）",
        "name": "model_f1_compare",
        "section": "B",
        "tags": ["模型评估"],
        "summary": "比较多个模型的整体分类效果（F1 兼顾 precision 与 recall）。",
        "how": [
            "柱越高表示在测试窗口更稳定；差距很小通常说明特征决定了上限。",
        ],
    },
    # C. 推荐评估
    "fig_hitk_curve.png": {
        "title": "Hit@K 对比曲线（多策略）",
        "name": "hitk_compare",
        "section": "C",
        "tags": ["推荐评估", "对比实验"],
        "summary": "对比不同推荐策略的命中率：看 model 是否明显高于 random，以及与 popular_train 的差距。",
        "how": [
            "横轴是 K（推荐列表长度），纵轴是 Hit@K（测试窗内是否命中过至少 1 道最终 AC 的题）。",
            "重点：model vs random（证明有效）；popular_train 是强基线（热门题）参照。",
        ],
        "tips": [
            "growth 策略可能牺牲部分命中率以换取更适度的学习题目，可结合难度分布图解释。",
        ],
    },
    "fig_reco_difficulty_hist.png": {
        "title": "推荐题难度分布（单用户案例）",
        "name": "reco_difficulty_hist",
        "section": "C",
        "tags": ["推荐解释", "成长带"],
        "summary": "检查推荐列表的难度结构是否“不过易也不过难”。",
        "how": [
            "如果全部偏低：缺挑战；全部偏高：命中与可学习性都差。",
        ],
    },
    "fig_reco_coverage.png": {
        "title": "推荐集中度与覆盖率（Top20 题被推荐次数）",
        "name": "reco_coverage",
        "section": "C",
        "tags": ["多样性", "同质化"],
        "summary": "检查是否总推荐少数热门题（同质化）；标题中 coverage 越高说明覆盖越广。",
        "how": [
            "Top20 柱子越集中且越高，说明推荐更同质化；coverage 越低说明推荐范围更窄。",
        ],
    },
}


def _html_ul(items: list[str]) -> str:
    if not items:
        return ""
    return "<ul style='margin:8px 0 0; padding-left:18px; font-size:0.9rem; color:var(--text-muted)'>" + "".join(
        f"<li>{html.escape(x)}</li>" for x in items if str(x).strip()
    ) + "</ul>"


def get_fig_info(filename: str) -> dict[str, object]:
    if filename in FIG_INFO:
        info = dict(FIG_INFO[filename])
    else:
        info = {}

    # Common defaults
    info.setdefault("title", filename)
    info.setdefault("name", filename.removeprefix("fig_").removesuffix(".png"))
    info.setdefault("section", "Z")
    info.setdefault("tags", [])
    info.setdefault("summary", "图表：用于展示训练数据、模型评估或推荐效果。")
    info.setdefault("how", [])
    info.setdefault("tips", [])

    # Confusion matrix family
    if filename.startswith("fig_cm_") or filename.startswith("fig_confusion_"):
        name = filename.removesuffix(".png").removeprefix("fig_cm_").removeprefix("fig_confusion_")
        title_map = {
            "logreg": "逻辑回归",
            "tree": "决策树",
            "svm_linear": "线性 SVM",
            "svm_or_knn": "SVM/KNN（对比）",
        }
        model_name = title_map.get(name, name)
        info.update(
            {
                "title": f"混淆矩阵：{model_name}",
                "name": f"cm_{name}",
                "section": "B",
                "tags": ["模型评估", "误差分析"],
                "summary": "把 AC 当作正类，拆解 TP/FP/FN/TN，解释 Precision/Recall 的来源。",
                "how": [
                    "对角线越高越好：左上=真负（预测未AC且确实未AC），右下=真正（预测AC且确实AC）。",
                    "右上=假正（误报AC，Precision 下降），左下=假负（漏报AC，Recall 下降）。",
                ],
                "tips": [
                    "假负多：模型偏保守；假正多：模型偏乐观。可通过特征/阈值/模型调整改善。",
                ],
            }
        )

    # strict vs leaky comparison figures
    if filename.startswith("fig_compare_"):
        key = filename.removesuffix(".png").removeprefix("fig_compare_")
        title_map = {
            "hitk": "Hit@K 对比（strict vs leaky）",
            "precisionk": "Precision@K 对比（strict vs leaky）",
            "roc": "ROC 曲线对比（strict vs leaky）",
            "pr": "PR 曲线对比（strict vs leaky）",
            "calibration": "校准曲线对比（strict vs leaky）",
        }
        info.update(
            {
                "title": title_map.get(key, f"对比图：{key}"),
                "name": f"compare_{key}",
                "section": "D",
                "tags": ["对比实验", "无泄漏验证"],
                "summary": "对比 strict（可部署口径）与 leaky（看未来口径）；若 leaky 明显更高则过去评估失真。",
                "how": [
                    "strict 更接近上线真实效果；报告结论应以 strict 为准。",
                    "ROC/PR 看排序能力；校准曲线看概率是否可信（是否过于乐观/保守）。",
                ],
            }
        )

    # Recommendation evaluation figure defaults
    if "reco_" in filename or "hitk" in filename:
        info.setdefault("section", "C")

    # Data distribution defaults
    if any(x in filename for x in ("_hist", "_dist", "vs_ac", "acrate")):
        info.setdefault("section", "A")

    # Friendly fallback explanations by pattern (only when not provided)
    if not info.get("how"):
        if any(x in filename for x in ("_hist", "_dist")):
            info["summary"] = info.get("summary") or "分布图：用于检查数据是否符合常识，以及是否存在异常集中/长尾。"
            info["how"] = [
                "先看横轴变量的取值范围（是否有异常值/不可能的取值）。",
                "再看纵轴数量分布（是否长尾、是否异常尖峰、是否过度集中）。",
            ]
        elif "vs_ac" in filename or "acrate" in filename:
            info["summary"] = info.get("summary") or "相关性图：用于查看某个因素与 AC 率的关系，验证特征是否有信息量。"
            info["how"] = [
                "横轴是特征（难度/尝试次数/语言/标签等），纵轴通常是 AC 率（0~1）。",
                "趋势是否可解释：难度升高通过率下降；不同语言/标签存在差异。",
            ]
        elif "coverage" in filename:
            info["summary"] = info.get("summary") or "覆盖率/集中度图：用于判断推荐是否同质化（是否总推荐少数题）。"
            info["how"] = [
                "Top20 柱越高且越集中，说明推荐越同质化；coverage 越高表示推荐更分散。",
            ]

    return info


FIG_SECTION_INFO: dict[str, dict[str, str]] = {
    "A": {
        "title": "A. 数据层（训练前）",
        "desc": "先证明数据与口径合理：分布符合常识、特征与 AC 率有可解释关联。",
    },
    "B": {
        "title": "B. 训练层（训练后）",
        "desc": "再展示模型效果与错误类型：F1 对比 + 混淆矩阵解释 precision/recall。",
    },
    "C": {
        "title": "C. 推荐评估（Top‑K）",
        "desc": "最后展示推荐效果：多策略对比（model / popular / random），并用案例与覆盖率解释推荐形态。",
    },
    "D": {
        "title": "D. 严格无泄漏对比（strict vs leaky）",
        "desc": "用于证明评估不失真：leaky 看未来会抬高指标，strict 更接近真实可部署效果。",
    },
    "Z": {"title": "其他图表", "desc": "未归类图表。"},
}

FIG_CANONICAL: dict[str, str] = {
    # duplicates from older scripts
    "fig_confusion_logreg.png": "fig_cm_logreg.png",
    "fig_confusion_tree.png": "fig_cm_tree.png",
    "fig_confusion_svm_linear.png": "fig_cm_svm_linear.png",
}


# ----------------------------------------------------------------------------
# 共享 HTML 头部/尾部 (含 Dark Mode JS)
# ----------------------------------------------------------------------------
HTML_HEAD = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OJ 推荐系统</title>
  <link rel="stylesheet" href="/static/style.css">
  <script>
    // 深色模式逻辑
    (function() {
      const saved = localStorage.getItem('theme');
      const sys = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (saved === 'dark' || (!saved && sys)) {
        document.documentElement.classList.add('dark-mode');
      } else {
        document.documentElement.classList.add('light-mode');
      }
    })();

    function toggleTheme() {
      const html = document.documentElement;
      if (html.classList.contains('dark-mode')) {
        html.classList.remove('dark-mode');
        html.classList.add('light-mode');
        localStorage.setItem('theme', 'light');
      } else {
        html.classList.remove('light-mode');
        html.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark');
      }
    }
  </script>
</head>
<body>
  <div class="container">
"""

HTML_HEADER_NAV = """
    <header>
      <div style="display:flex; align-items:center; gap:12px;">
         <div style="font-size:24px;">🧠</div>
         <div>
            <h1 style="margin:0; font-size:1.5rem;">OJ 数据分析与推荐</h1>
         </div>
      </div>
      <div style="display:flex; align-items:center; gap:20px;">
        <nav style="display:flex; gap:15px;">
           <a href="/">首页大盘</a>
           <a href="/student">学生画像</a>
           <a href="/custom">自定义推荐</a>
        </nav>
        <button class="theme-toggle" onclick="toggleTheme()" title="切换深色/浅色模式">
           ◑
        </button>
      </div>
    </header>
"""

HTML_FOOTER = """
    <footer style="margin-top:40px; padding-top:20px; border-top:1px solid var(--border); text-align:center; color:var(--text-muted); font-size:0.85rem;">
        &copy; 2024 Intelligent OJ Recommender System
    </footer>
  </div> <!-- end container -->
</body>
</html>
"""


def html_head(page: str) -> str:
    page = (page or "").strip().lower() or "page"
    safe = "".join(ch for ch in page if (ch.isalnum() or ch in {"-", "_"}))
    if not safe:
        safe = "page"
    return HTML_HEAD.replace("<body>", f'<body class="page page-{safe}">', 1)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        global RECO
        p = urlparse(self.path)

        # 静态 CSS
        if p.path == "/static/style.css":
            self._send(200, STYLE_CSS.encode("utf-8"), "text/css; charset=utf-8")
            return

        # API: Student Data
        if p.path == "/api/student":
            if RECO is None:
                try:
                    RECO = Recommender()
                except Exception as e:
                    self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                    return

            q = parse_qs(p.query or "")
            try:
                user_id = int(q.get("user_id", ["1"])[0])
                pct = float(q.get("pct", ["0.5"])[0])
                k = int(q.get("k", ["10"])[0])
                min_p = float(q.get("min_p", ["0.4"])[0])
                max_p = float(q.get("max_p", ["0.7"])[0])
            except Exception:
                self._send(400, "参数错误".encode("utf-8"), "text/plain; charset=utf-8")
                return

            try:
                payload = RECO.student_dashboard_payload(
                    user_id=user_id,
                    cutoff_pct=pct,
                    k=k,
                    min_p=min_p,
                    max_p=max_p,
                )
            except Exception as e:
                self._send(500, f"生成失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                return

            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return

        # Page: Student Dashboard
        if p.path == "/student":
            if RECO is None:
                try:
                    RECO = Recommender()
                except Exception as e:
                    self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                    return

            user_ids = sorted([int(x) for x in RECO._subs_by_user.keys()]) if getattr(RECO, "_subs_by_user", None) else []
            if not user_ids:
                user_ids = [1]
            default_uid = 1 if 1 in user_ids else user_ids[0]
            user_opts = "".join(
                f'<option value="{uid}"{" selected" if uid == default_uid else ""}>{uid}</option>' for uid in user_ids
            )

            body = f"""
{html_head("student")}
{HTML_HEADER_NAV}

    <div class="grid">
        <!-- Controls -->
        <div class="card">
          <h3>👤 学生筛选</h3>
          <div style="margin-bottom:1rem;">
            <label for="user_id">选择学生（user_id）</label>
            <select id="user_id">{user_opts}</select>
            <div class="help">提示：可在下拉框中键入数字快速定位；也可用下面输入框过滤列表。</div>
            <input id="user_filter" type="text" placeholder="过滤 user_id（例如：12 / 1001 / 42）" style="margin-top:10px;">
            <div style="display:flex; gap:10px; margin-top:10px;">
                <button id="refresh_btn" class="btn-primary" style="flex:1;">分析</button>
            </div>
          </div>
          <div class="muted" id="status">准备就绪</div>
        </div>

        <div class="card">
          <h3>⚙️ 推荐参数</h3>
          <div class="subgrid">
            <div>
              <label>历史切片比例（Cutoff Pct）</label>
              <div class="row">
                <input id="pct" type="range" min="0" max="1" step="0.01" value="0.50">
                <output id="pct_out">0.50</output>
              </div>
              <div class="help">把该学生提交序列的前 <span class="mono">pct</span> 作为“历史”，用来重算画像与推荐；越大越接近“当前”。</div>
            </div>
            <div>
              <label>推荐数量（Top K）</label>
              <div class="row">
                <input id="k" type="range" min="1" max="50" step="1" value="10">
                <output id="k_out">10</output>
              </div>
                    <div class="help">输出 Top‑K 推荐列表长度。若成长带内题不足，会自动用“最接近成长带”的题补齐。</div>
            </div>
            <div>
              <label>成长带成功率下限（min_p）</label>
              <div class="row">
                <input id="min_p" type="range" min="0" max="1" step="0.01" value="0.40">
                <output id="min_p_out">0.40</output>
              </div>
              <div class="help">只优先推荐预测通过率 <span class="mono">P(AC)</span> ≥ min_p 的题，避免太难。</div>
            </div>
            <div>
              <label>成长带成功率上限（max_p）</label>
              <div class="row">
                <input id="max_p" type="range" min="0" max="1" step="0.01" value="0.70">
                <output id="max_p_out">0.70</output>
              </div>
              <div class="help">只优先推荐 <span class="mono">P(AC)</span> ≤ max_p 的题，避免“太容易刷分”。</div>
            </div>
          </div>
        </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">📊 用户画像快照（按历史重算）</h3>
      <div class="help">说明：下列画像/偏好仅使用 cutoff 之前的历史提交重算（不看未来），避免展示“看起来很准但其实把未来算进来了”的时间泄漏。</div>
      <div class="muted" id="meta" style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:10px;">
        请点击“分析”加载数据...
      </div>
    </div>

    <div class="grid">
        <div class="card">
          <h3>时间轴与成长轨迹</h3>
          <div class="chart-container"><img id="img_timeline" class="viz-img" alt="Timeline"></div>
          <div class="help">红点=未 AC，绿点=AC；蓝色竖线=历史切片点；蓝色星标=推荐题难度（落在切片点之后）。</div>
        </div>
        <div class="card">
          <h3>能力雷达对比</h3>
          <div class="chart-container"><img id="img_radar" class="viz-img" alt="Radar"></div>
          <div class="help">对比“历史行为偏好”与“推荐列表”在语言/标签维度上的形状差异，用于解释推荐是否贴合该学生。</div>
        </div>
    </div>

    <div class="card">
        <h3>难度阶梯与 P(AC)</h3>
        <div class="chart-container"><img id="img_ladder" class="viz-img" alt="Ladder"></div>
        <div class="help">按难度阶梯展示题目预测通过率：通常期望“成长带（0.4~0.7）”附近有较多可选题。</div>
    </div>

    <div class="card">
      <h3>📋 推荐题目列表</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th width="50">#</th>
              <th width="80">ID</th>
              <th>题目名称</th>
              <th width="80">难度</th>
              <th>核心标签</th>
              <th width="100">语言</th>
              <th width="100">预测通过率</th>
              <th width="80">ZPD</th>
            </tr>
          </thead>
          <tbody id="reco_rows"></tbody>
        </table>
      </div>
    </div>

<script>
function bindOut(id, outId, fmt) {{
  const el = document.getElementById(id);
  const out = document.getElementById(outId);
  const update = () => out.textContent = fmt(el.value);
  el.addEventListener("input", update);
  update();
}}
bindOut("pct","pct_out",(v)=>Number(v).toFixed(2));
bindOut("k","k_out",(v)=>String(v));
bindOut("min_p","min_p_out",(v)=>Number(v).toFixed(2));
bindOut("max_p","max_p_out",(v)=>Number(v).toFixed(2));

function esc(s) {{
  return String(s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}}

async function refresh() {{
  const status = document.getElementById("status");
  const btn = document.getElementById("refresh_btn");
  status.textContent = "正在计算...";
  btn.disabled = true;
  btn.textContent = "计算中...";

  const user_id = String(document.getElementById("user_id").value || "").trim() || "1";
  const pct = document.getElementById("pct").value;
  const k = document.getElementById("k").value;
  const min_p = document.getElementById("min_p").value;
  const max_p = document.getElementById("max_p").value;

  const url = `/api/student?user_id=${{encodeURIComponent(user_id)}}&pct=${{encodeURIComponent(pct)}}&k=${{encodeURIComponent(k)}}&min_p=${{encodeURIComponent(min_p)}}&max_p=${{encodeURIComponent(max_p)}}`;

  try {{
    const resp = await fetch(url);
    const text = await resp.text();
    if(!resp.ok) throw new Error(text);
    const data = JSON.parse(text);

    // Render Meta
    const m = data.meta || {{}};
    document.getElementById("meta").innerHTML =
      `<div><span class="muted">切片点 submission_id（时间近似）：</span> <b>${{m.cutoff_submission_id}}</b></div>` +
      `<div><span class="muted">历史提交数：</span> <b>${{m.hist_submissions}}</b></div>` +
      `<div><span class="muted">历史已 AC 题数：</span> <b>${{m.hist_solved}}</b></div>` +
      `<div><span class="muted">能力 level（0~1）：</span> <b>${{Number(m.level).toFixed(3)}}</b></div>` +
      `<div><span class="muted">坚持度 perseverance（0~1）：</span> <b>${{Number(m.perseverance).toFixed(3)}}</b></div>` +
      `<div><span class="muted">历史常用语言：</span> <b>${{esc(m.top_language)}}</b></div>` +
      `<div><span class="muted">成长带（P(AC)）：</span> <b>[${{Number((m.zpd||[])[0] ?? 0).toFixed(2)}}, ${{Number((m.zpd||[])[1] ?? 0).toFixed(2)}}]</b></div>`;

    // Render Images
    const imgs = data.images || {{}};
    document.getElementById("img_timeline").src = "data:image/png;base64," + (imgs.timeline_scatter || "");
    document.getElementById("img_radar").src = "data:image/png;base64," + (imgs.radar_compare || "");
    document.getElementById("img_ladder").src = "data:image/png;base64," + (imgs.difficulty_ladder || "");

    // Render Table
    const rows = data.recommendations || [];
    const tbody = document.getElementById("reco_rows");
    tbody.innerHTML = rows.map(r => {{
      const p = Number(r.p_ac);
      const scoreClass = p >= 0.7 ? "color:var(--success)" : (p >= 0.4 ? "color:var(--warning)" : "color:var(--danger)");
      return `<tr>` +
        `<td>${{r.rank}}</td>` +
        `<td style="font-family:monospace" class="muted">#${{r.problem_id}}</td>` +
        `<td>${{esc(r.title)}}</td>` +
        `<td><span class="pill">${{r.difficulty}}</span></td>` +
        `<td class="muted" style="font-size:0.85em">${{esc(r.tags)}}</td>` +
        `<td>${{esc(r.language)}}</td>` +
        `<td style="font-weight:700;${{scoreClass}}">${{p.toFixed(3)}}</td>` +
        `<td>${{r.in_growth_band ? "✅" : ""}}</td>` +
      `</tr>`;
    }}).join("");

    status.textContent = "计算完成";
  }} catch (e) {{
    status.textContent = "错误：" + (e.message || e);
  }} finally {{
    btn.disabled = false;
    btn.textContent = "分析";
  }}
}}

document.getElementById("refresh_btn").addEventListener("click", refresh);
document.getElementById("user_id").addEventListener("change", refresh);
["pct","k","min_p","max_p"].forEach(id => document.getElementById(id).addEventListener("change", refresh));

// dropdown filter
document.getElementById("user_filter").addEventListener("input", () => {{
  const q = String(document.getElementById("user_filter").value || "").trim();
  const sel = document.getElementById("user_id");
  const opts = Array.from(sel.options || []);
  for (const o of opts) {{
    o.hidden = !!q && !String(o.value).includes(q);
  }}
  // if current selection hidden, jump to first visible
  const cur = sel.options[sel.selectedIndex];
  if (cur && cur.hidden) {{
    const first = opts.find(x => !x.hidden);
    if (first) sel.value = first.value;
  }}
}});
// Auto load on first view
refresh();
</script>
{HTML_FOOTER}
""".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

        # Serve Images
        if p.path.startswith("/reports/"):
            name = p.path.removeprefix("/reports/").lstrip("/")
            if "/" in name or ".." in name or not name.endswith(".png"):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            path = REPORTS_DIR / name
            if not path.exists():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            self._send(200, path.read_bytes(), "image/png")
            return

        # Page: Index (Visualizations)
        if p.path in {"/", "/index.html"}:
            figs = sorted([x.name for x in REPORTS_DIR.glob("fig_*.png")])
            figs_set = set(figs)

            # canonicalize (hide duplicates)
            canonical_figs: list[str] = []
            for fn in figs:
                if FIG_CANONICAL.get(fn, fn) != fn and FIG_CANONICAL.get(fn, fn) in figs_set:
                    continue
                canonical_figs.append(fn)

            def render_fig_card(fn: str) -> str:
                info = get_fig_info(fn)
                title = str(info.get("title") or fn)
                name = str(info.get("name") or fn.removeprefix("fig_").removesuffix(".png"))
                summary = str(info.get("summary") or "")
                section = str(info.get("section") or "Z")

                chip = "chip"
                if section == "A":
                    chip += " primary"
                elif section == "B":
                    chip += " success"
                elif section == "C":
                    chip += " warn"
                elif section == "D":
                    chip += " danger"

                return f"""
                <div class="card fig-card" data-fn="{html.escape(fn)}" data-title="{html.escape(title)}" data-section="{html.escape(section)}" data-name="{html.escape(name)}">
                  <div class="fig-meta" style="margin-bottom:6px">
                    <span class="{chip}">{html.escape(FIG_SECTION_INFO.get(section, FIG_SECTION_INFO['Z'])['title'].split('.')[0])}</span>
                    <span class="chip mono">{html.escape(name)}</span>
                  </div>
                  <h3>{html.escape(title)}</h3>
                  <div class="img-frame">
                    <button type="button" class="img-button" onclick="openFigModal('{html.escape(fn)}')" title="点击查看大图与解读">
                      <img src="/reports/{html.escape(fn)}" loading="lazy" alt="{html.escape(fn)}">
                    </button>
                  </div>
                  <div class="muted fig-summary">{html.escape(summary)}</div>
                </div>
                """

            # group by section
            sec_to_figs: dict[str, list[str]] = {k: [] for k in FIG_SECTION_INFO.keys()}
            for fn in canonical_figs:
                sec = str(get_fig_info(fn).get("section") or "Z")
                if sec not in sec_to_figs:
                    sec_to_figs[sec] = []
                sec_to_figs[sec].append(fn)

            section_order = ["A", "B", "C", "D", "Z"]
            sections_html = ""
            total = len(canonical_figs)
            if total == 0:
                sections_html = '<div class="card muted" style="text-align:center; padding:40px;">暂无图表文件，请先运行分析脚本生成 Reports/fig_*.png。</div>'
            else:
                for sec in section_order:
                    fns = sec_to_figs.get(sec, [])
                    if not fns:
                        continue
                    meta = FIG_SECTION_INFO.get(sec, FIG_SECTION_INFO["Z"])
                    sections_html += f"""
                    <section data-sec="{sec}" id="sec-{sec}">
                      <div class="section-header">
                        <div>
                          <h2>{html.escape(meta['title'])}</h2>
                          <div class="section-desc">{html.escape(meta['desc'])}</div>
                        </div>
                        <div class="chip mono">count: {len(fns)}</div>
                      </div>
                      <div class="grid">
                        {''.join(render_fig_card(fn) for fn in fns)}
                      </div>
                    </section>
                    """

            # build meta for modal rendering (json serializable)
            fig_meta: dict[str, dict[str, object]] = {}
            for fn in canonical_figs:
                info = get_fig_info(fn)
                fig_meta[fn] = {
                    "title": str(info.get("title") or fn),
                    "name": str(info.get("name") or fn.removeprefix("fig_").removesuffix(".png")),
                    "section": str(info.get("section") or "Z"),
                    "tags": [str(x) for x in (info.get("tags") or [])] if isinstance(info.get("tags"), list) else [],
                    "summary": str(info.get("summary") or ""),
                    "how": [str(x) for x in (info.get("how") or [])] if isinstance(info.get("how"), list) else [],
                    "tips": [str(x) for x in (info.get("tips") or [])] if isinstance(info.get("tips"), list) else [],
                }

            body = f"""
{html_head("home")}
{HTML_HEADER_NAV}
    <div class="card sticky">
      <div class="toolbar-simple">
        <div class="topline">
          <div>
            <h2 style="margin:0 0 6px">📈 首页大盘</h2>
            <div class="muted">搜索图表并点击图片查看解读（支持大图/要点/提示）。</div>
          </div>
          <div class="chip mono">总图数: {total}</div>
        </div>
        <div class="filters">
          <div>
            <label for="q">搜索（标题/短名/文件名）</label>
            <input id="q" type="text" placeholder="例如：difficulty / hitk / cm / calibration ...">
          </div>
          <details>
            <summary>筛选与跳转</summary>
            <div style="margin-top:10px">
              <div class="muted" style="margin-bottom:10px">
                命名约定：<span class="mono">fig_*</span> 图表；<span class="mono">fig_cm_*</span> 混淆矩阵；<span class="mono">fig_compare_*</span> strict vs leaky 对比。
              </div>
              <div class="muted" style="margin-bottom:10px">
                快速跳转：
                <a class="mono" href="#sec-A">A</a> /
                <a class="mono" href="#sec-B">B</a> /
                <a class="mono" href="#sec-C">C</a> /
                <a class="mono" href="#sec-D">D</a>
              </div>
              <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
                <span class="chip"><input id="fA" type="checkbox" checked style="margin-right:6px">A</span>
                <span class="chip"><input id="fB" type="checkbox" checked style="margin-right:6px">B</span>
                <span class="chip"><input id="fC" type="checkbox" checked style="margin-right:6px">C</span>
                <span class="chip"><input id="fD" type="checkbox" checked style="margin-right:6px">D</span>
                <span class="chip"><input id="fZ" type="checkbox" checked style="margin-right:6px">其他</span>
                <button class="btn btn-secondary btn-sm" type="button" onclick="resetFilters()">重置</button>
              </div>
            </div>
          </details>
        </div>
      </div>
    </div>

    <div id="sections">
      {sections_html}
    </div>

    <div id="fig_modal" class="modal-overlay hidden" role="dialog" aria-modal="true" aria-label="图表大图与解读">
      <div class="modal">
        <div class="modal-header">
          <div style="min-width:0">
            <div class="fig-meta" style="margin-bottom:6px">
              <span id="modal_chip" class="chip">Z</span>
              <span id="modal_name" class="chip mono"></span>
              <span id="modal_tags" style="display:flex; gap:6px; flex-wrap:wrap;"></span>
            </div>
            <h3 id="modal_title"></h3>
            <div id="modal_path" class="muted mono"></div>
          </div>
          <button class="modal-close" type="button" onclick="closeFigModal()" title="关闭">×</button>
        </div>
        <div class="modal-body">
          <div class="modal-left">
            <img id="modal_img" alt="figure">
          </div>
          <div class="modal-right">
            <div style="margin-bottom:14px">
              <h4>这张图的作用</h4>
              <div id="modal_summary" class="muted"></div>
            </div>
            <div style="margin-bottom:14px">
              <h4>这张图怎么看</h4>
              <ul id="modal_how" class="modal-ul"></ul>
            </div>
            <div id="modal_tips_block" style="margin-bottom:14px">
              <h4>提示 / 常见误读</h4>
              <ul id="modal_tips" class="modal-ul"></ul>
            </div>
            <div class="modal-actions">
              <a id="modal_open" class="btn btn-secondary btn-sm" target="_blank" rel="noreferrer">新窗口打开</a>
              <a id="modal_download" class="btn btn-secondary btn-sm" download>下载图片</a>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      window.FIG_META = {json.dumps(fig_meta, ensure_ascii=False, separators=(",", ":"))};

      function copyText(t) {{
        try {{
          navigator.clipboard.writeText(t);
        }} catch (e) {{
          const ta = document.createElement('textarea');
          ta.value = t;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          ta.remove();
        }}
      }}

      function resetFilters() {{
        document.getElementById('q').value = '';
        ['fA','fB','fC','fD','fZ'].forEach(id => document.getElementById(id).checked = true);
        applyFilters();
      }}

      function applyFilters() {{
        const q = (document.getElementById('q').value || '').trim().toLowerCase();
        const allow = {{
          'A': document.getElementById('fA').checked,
          'B': document.getElementById('fB').checked,
          'C': document.getElementById('fC').checked,
          'D': document.getElementById('fD').checked,
          'Z': document.getElementById('fZ').checked,
        }};

        const cards = Array.from(document.querySelectorAll('.fig-card'));
        for (const c of cards) {{
          const sec = c.getAttribute('data-section') || 'Z';
          const text = (c.getAttribute('data-title') || '') + ' ' + (c.getAttribute('data-name') || '') + ' ' + (c.getAttribute('data-fn') || '');
          const ok = (allow[sec] ?? true) && (!q || text.toLowerCase().includes(q));
          c.classList.toggle('hidden', !ok);
        }}

        // hide empty sections
        const sections = Array.from(document.querySelectorAll('#sections section'));
        for (const s of sections) {{
          const any = s.querySelector('.fig-card:not(.hidden)');
          s.classList.toggle('hidden', !any);
        }}
      }}

      document.getElementById('q').addEventListener('input', applyFilters);
      ['fA','fB','fC','fD','fZ'].forEach(id => document.getElementById(id).addEventListener('change', applyFilters));
      applyFilters();

      function _escHtml(s) {{
        return String(s || "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
      }}
      function _chipClass(section) {{
        if (section === 'A') return 'chip primary';
        if (section === 'B') return 'chip success';
        if (section === 'C') return 'chip warn';
        if (section === 'D') return 'chip danger';
        return 'chip';
      }}
      function openFigModal(fn) {{
        const meta = (window.FIG_META || {{}})[fn] || {{}};
        const title = meta.title || fn;
        const name = meta.name || fn.replace(/^fig_/, '').replace(/\\.png$/, '');
        const section = meta.section || 'Z';
        const tags = Array.isArray(meta.tags) ? meta.tags : [];
        const how = Array.isArray(meta.how) ? meta.how : [];
        const tips = Array.isArray(meta.tips) ? meta.tips : [];

        const overlay = document.getElementById('fig_modal');
        const chip = document.getElementById('modal_chip');
        chip.className = _chipClass(section);
        chip.textContent = section;

        document.getElementById('modal_name').textContent = name;
        document.getElementById('modal_title').textContent = title;
        document.getElementById('modal_path').textContent = `Reports/${{fn}}`;
        document.getElementById('modal_img').src = `/reports/${{fn}}`;
        document.getElementById('modal_summary').textContent = meta.summary || '（暂无说明）';

        const tagWrap = document.getElementById('modal_tags');
        tagWrap.innerHTML = tags.slice(0, 8).map(t => `<span class="chip">${{_escHtml(t)}}</span>`).join('');

        const howEl = document.getElementById('modal_how');
        howEl.innerHTML = (how.length ? how : ['（暂无具体解读条目）']).map(x => `<li>${{_escHtml(x)}}</li>`).join('');

        const tipsBlock = document.getElementById('modal_tips_block');
        const tipsEl = document.getElementById('modal_tips');
        tipsEl.innerHTML = tips.map(x => `<li>${{_escHtml(x)}}</li>`).join('');
        tipsBlock.style.display = tips.length ? '' : 'none';

        const openA = document.getElementById('modal_open');
        const dlA = document.getElementById('modal_download');
        openA.href = `/reports/${{fn}}`;
        dlA.href = `/reports/${{fn}}`;

        overlay.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
      }}
      function closeFigModal() {{
        const overlay = document.getElementById('fig_modal');
        overlay.classList.add('hidden');
        document.body.style.overflow = '';
      }}
      document.getElementById('fig_modal').addEventListener('click', (e) => {{
        if (e.target && e.target.id === 'fig_modal') closeFigModal();
      }});
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closeFigModal();
      }});
    </script>

{HTML_FOOTER}
""".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

        # Page: Custom Recommendation Form
        if p.path == "/custom":
            if RECO is None:
                try:
                    RECO = Recommender()
                except Exception as e:
                    self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                    return

            tag_opts = "".join(f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in RECO.tag_names)
            lang_opts = "".join(f'<option value="{html.escape(l)}">{html.escape(l)}</option>' for l in RECO.lang_names)

            body = f"""
{html_head("custom")}
{HTML_HEADER_NAV}

    <form method="post" action="/custom">
        <div class="grid">
            <!-- Left Column: User Profile -->
            <div>
                <div class="card">
                    <h3>🎭 模拟用户画像</h3>
                    <div style="margin-bottom:1.2rem;">
                        <label>能力水平（level）</label>
                        <div class="row">
                            <span class="muted" style="font-size:0.8rem">新手</span>
                            <input id="level" name="level" type="range" min="0" max="1" step="0.01" value="0.50">
                            <span class="muted" style="font-size:0.8rem">专家</span>
                            <output id="level_out">0.50</output>
                        </div>
                        <div class="help">0~1 归一化能力值：越大越强（这里只是模拟输入，用于体验推荐逻辑）。</div>
                    </div>
                    <div style="margin-bottom:1.2rem;">
                        <label>坚持度（perseverance）</label>
                        <div class="row">
                             <span class="muted" style="font-size:0.8rem">易弃</span>
                            <input id="perseverance" name="perseverance" type="range" min="0" max="1" step="0.01" value="0.60">
                             <span class="muted" style="font-size:0.8rem">坚韧</span>
                            <output id="perseverance_out">0.60</output>
                        </div>
                        <div class="help">0~1 归一化坚持度：越大表示更愿意在同题多次尝试（影响“需要重试”的题是否适合）。</div>
                    </div>
                    <div>
                        <label>当前尝试次数（attempt_no）</label>
                        <div class="row">
                            <input id="attempt_no" name="attempt_no" type="range" min="1" max="10" step="1" value="1">
                            <output id="attempt_no_out">1</output>
                        </div>
                        <div class="help">同一题第几次尝试：一般尝试次数越多，后续 AC 概率越高（模型会学到这个趋势）。</div>
                    </div>
                </div>

                <div class="card">
                    <h3>💻 技术偏好</h3>
                    <div class="subgrid">
                        <div>
                            <label>首选语言</label>
                            <select id="lang_top" name="lang_top">{lang_opts}</select>
                            <div class="help">用于模拟“最常用/最擅长”的语言偏好。</div>
                        </div>
                        <div>
                            <label>语言权重</label>
                            <div class="row">
                                <input id="lang_strength" name="lang_strength" type="range" min="0.5" max="0.95" step="0.01" value="0.70">
                                <output id="lang_strength_out">0.70</output>
                            </div>
                            <div class="help">越高表示越偏向首选语言；剩余权重会平均分给其他语言。</div>
                        </div>
                    </div>

                    <div style="margin-top:1.5rem">
                         <label>感兴趣的题型标签（多选）</label>
                         <select id="tag_selected" name="tag_selected" multiple size="8" class="multiselect">{tag_opts}</select>
                         <div class="help">按住 Ctrl/⌘ 可多选；不选表示“无明显题型偏好”（会均匀分配）。</div>
                         <div class="row" style="margin-top:10px">
                            <span style="font-size:0.8rem" class="muted">标签权重</span>
                            <input id="tag_strength" name="tag_strength" type="range" min="0.5" max="0.95" step="0.01" value="0.70">
                            <output id="tag_strength_out">0.70</output>
                         </div>
                         <div class="help">越高表示越集中在所选标签；越低表示更“均衡探索”。</div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Algorithm Config -->
            <div>
                <div class="card">
                    <h3>⚙️ 推荐配置</h3>
                    <div style="margin-bottom:1.2rem;">
                        <label>推荐数量（Top K）</label>
                        <div class="row">
                            <input id="k" name="k" type="range" min="1" max="50" step="1" value="10">
                            <output id="k_out">10</output>
                        </div>
                        <div class="help">输出 Top‑K 推荐列表长度。</div>
                    </div>

                    <label>成长带：成功率区间（ZPD）</label>
                    <div style="background:var(--bg-body); padding:10px; border-radius:8px; border:1px solid var(--border);">
                        <div style="margin-bottom:8px">
                            <label style="font-size:0.8rem">下限（min_p）</label>
                            <div class="row">
                                <input id="min_p" name="min_p" type="range" min="0" max="1" step="0.01" value="0.40">
                                <output id="min_p_out">0.40</output>
                            </div>
                        </div>
                        <div>
                            <label style="font-size:0.8rem">上限（max_p）</label>
                            <div class="row">
                                <input id="max_p" name="max_p" type="range" min="0" max="1" step="0.01" value="0.70">
                                <output id="max_p_out">0.70</output>
                            </div>
                        </div>
                    </div>
                    <div class="help">优先推荐预测通过率 <span class="mono">P(AC)</span> 落在 [min_p, max_p] 的题：既不过难也不过易；若带内题不足会用“最接近成长带”的题补齐。</div>

                    <div style="margin-top:1.5rem">
                        <label>融合策略（语言偏好如何作用）</label>
                        <select name="mode">
                            <option value="expected" selected>期望值（标准：按语言权重加权）</option>
                            <option value="best">强项优先（激进：仅看最擅长语言）</option>
                        </select>
                        <div class="help">期望值：同时考虑多语言偏好；强项优先：更像“用最强语言去做题”的上限视角。</div>
                    </div>
                </div>

                <div class="card">
                    <h3>🚀 执行操作</h3>
                    <div class="actions" style="border:none; margin:0; padding:0; flex-direction:column;">
                        <button type="submit" style="width:100%; padding:1rem; font-size:1.1rem;">生成推荐列表</button>
                        <div style="display:flex; gap:10px; width:100%;">
                            <button type="button" class="btn-secondary" id="btn_random" style="flex:1">🎲 随机参数</button>
                            <button type="button" class="btn-secondary" id="btn_reset" style="flex:1">🔄 重置</button>
                        </div>
                    </div>

                    <details style="margin-top:15px">
                        <summary>高级: JSON 注入</summary>
                        <div style="margin-top:10px">
                             <label>语言权重 JSON（会覆盖上方“首选语言/语言权重”）</label>
                             <textarea id="lang_json" name="lang_json" rows="2" placeholder='{{"Python":0.9}}'></textarea>
                             <label style="margin-top:10px">标签权重 JSON（会覆盖上方“多选标签/标签权重”）</label>
                             <textarea id="tag_json" name="tag_json" rows="2" placeholder='{{"dp":0.5}}'></textarea>
                        </div>
                    </details>
                </div>
            </div>
        </div>
    </form>

<script>
function bind(id) {{
    const el = document.getElementById(id);
    const out = document.getElementById(id+"_out");
    if(el && out) {{
        el.addEventListener("input", ()=>out.textContent = Number(el.value).toFixed(el.step.includes(".")?2:0));
    }}
}}
["level","perseverance","attempt_no","k","min_p","max_p","lang_strength","tag_strength"].forEach(bind);

document.getElementById("min_p").addEventListener("change", function(){{
    const max = document.getElementById("max_p");
    if(Number(this.value) > Number(max.value)) max.value = this.value;
    max.dispatchEvent(new Event("input"));
}});

document.getElementById("btn_random").addEventListener("click", ()=>{{
    const r = (min,max) => (Math.random()*(max-min)+min).toFixed(2);
    document.getElementById("level").value = r(0.1, 0.9);
    document.getElementById("perseverance").value = r(0.1, 0.9);
    document.getElementById("k").value = Math.floor(Math.random()*15)+5;
    document.getElementById("min_p").value = r(0.3, 0.5);
    document.getElementById("max_p").value = r(0.6, 0.85);

    // Trigger updates
    document.querySelectorAll("input[type=range]").forEach(e => e.dispatchEvent(new Event("input")));
}});

document.getElementById("btn_reset").addEventListener("click", ()=>{{
    document.querySelector("form").reset();
    setTimeout(()=>document.querySelectorAll("input[type=range]").forEach(e => e.dispatchEvent(new Event("input"))), 10);
}});
</script>
{HTML_FOOTER}
""".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        p = urlparse(self.path)
        if p.path != "/custom":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        global RECO
        if RECO is None:
            try:
                RECO = Recommender()
            except Exception as e:
                self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                return

        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        q = parse_qs(raw)

        def get1(name: str, default: str = "") -> str:
            v = q.get(name, [])
            return v[0] if v else default

        def getlist(name: str) -> list[str]:
            return [x for x in q.get(name, []) if x]

        try:
            level = float(get1("level", "0.5"))
            perseverance = float(get1("perseverance", "0.6"))
            attempt_no = int(get1("attempt_no", "1"))
            k = int(get1("k", "10"))
            min_p = float(get1("min_p", "0.4"))
            max_p = float(get1("max_p", "0.7"))
        except Exception:
            self._send(400, "参数格式错误".encode("utf-8"), "text/plain; charset=utf-8")
            return

        mode = get1("mode", "expected").strip()

        lang_json = get1("lang_json", "").strip()
        if lang_json:
            lang_pref = parse_json_dict(lang_json)
        else:
            top = get1("lang_top", RECO.lang_names[0] if RECO.lang_names else "")
            try:
                strength = float(get1("lang_strength", "0.70"))
            except Exception:
                strength = 0.70
            strength = max(0.50, min(0.95, strength))
            lang_pref = {l: 0.0 for l in RECO.lang_names}
            if top in lang_pref:
                lang_pref[top] = strength
            rest = [l for l in RECO.lang_names if l != top]
            for l in rest:
                lang_pref[l] = lang_pref.get(l, 0.0) + (1.0 - strength) / max(1, len(rest))

        tag_json = get1("tag_json", "").strip()
        if tag_json:
            tag_pref = parse_json_dict(tag_json)
        else:
            chosen = [t for t in getlist("tag_selected") if t in set(RECO.tag_names)]
            try:
                strength = float(get1("tag_strength", "0.70"))
            except Exception:
                strength = 0.70
            strength = max(0.50, min(0.95, strength))
            tag_pref = {t: 0.0 for t in RECO.tag_names}
            if chosen:
                for t in chosen:
                    tag_pref[t] = strength / len(chosen)
            rest = [t for t in RECO.tag_names if t not in set(chosen)]
            for t in rest:
                tag_pref[t] = tag_pref.get(t, 0.0) + (1.0 - strength) / max(1, len(rest))

        try:
            out, img_b64 = RECO.recommend(
                level=level,
                perseverance=perseverance,
                lang_pref=lang_pref,
                tag_pref=tag_pref,
                k=k,
                attempt_no=attempt_no,
                min_p=min_p,
                max_p=max_p,
                mode=("best" if mode == "best" else "expected"),
            )
        except Exception as e:
            self._send(500, f"推荐生成失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
            return

        rows = []
        for i, r in out.iterrows():
            score_val = float(r['score'])
            score_style = "color:var(--success)" if score_val > 0.6 else (
                "color:var(--warning)" if score_val > 0.4 else "color:var(--danger)")

            rows.append(
                "<tr>"
                f"<td class='muted'>#{int(r['problem_id'])}</td>"
                f"<td style='font-weight:600'>{html.escape(str(r.get('title') or ''))}</td>"
                f"<td><span class='pill'>{int(r['difficulty_filled'])}</span></td>"
                f"<td class='muted'>{html.escape(str(r.get('tags') or ''))}</td>"
                f"<td>{html.escape(str(r.get('language') or ''))}</td>"
                f"<td style='font-weight:700;{score_style}'>{score_val:.3f}</td>"
                "</tr>"
            )

        lang_top = ""
        if lang_pref:
            try:
                lang_top = max(lang_pref.items(), key=lambda kv: float(kv[1]))[0]
            except Exception:
                lang_top = ""
        top_tags = ""
        if tag_pref:
            try:
                top_tags = "、".join([k for k, v in sorted(tag_pref.items(), key=lambda kv: float(kv[1]), reverse=True)[:3] if float(v) > 0])
            except Exception:
                top_tags = ""
        mode_cn = "强项优先（取最大）" if mode == "best" else "期望值（按权重加权）"

        body = f"""
{html_head("custom")}
{HTML_HEADER_NAV}

    <div style="margin-bottom:20px">
       <a href="/custom" class="btn btn-secondary">&larr; 调整参数</a>
    </div>

    <div class="grid">
        <div class="card">
            <h3>⚙️ 推荐上下文</h3>
            <div class="help">说明：本页仅做“参数 → 推荐结果”的可解释展示。模型来自离线训练产物 <span class="mono">Models/pipeline_logreg.joblib</span>，Web 端只加载并推理。</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.9rem; color:var(--text-muted); margin-top:10px;">
                <div>能力 level（0~1）：<b style="color:var(--text-main)">{level:.2f}</b></div>
                <div>坚持度 perseverance（0~1）：<b style="color:var(--text-main)">{perseverance:.2f}</b></div>
                <div>尝试次数 attempt_no：<b style="color:var(--text-main)">{attempt_no}</b></div>
                <div>推荐数量 Top‑K：<b style="color:var(--text-main)">{k}</b></div>
                <div>成长带 P(AC)：<b style="color:var(--text-main)">[{min_p:.2f}, {max_p:.2f}]</b></div>
                <div>融合策略：<b style="color:var(--text-main)">{html.escape(mode_cn)}</b></div>
                <div>首选语言（权重最高）：<b style="color:var(--text-main)">{html.escape(lang_top)}</b></div>
                <div>主要标签（Top3）：<b style="color:var(--text-main)">{html.escape(top_tags) if top_tags else "（未指定）"}</b></div>
            </div>
            <ul class="modal-ul" style="margin-top:10px">
              <li>成长带（ZPD）：优先挑选预测通过率在区间内的题，兼顾“可命中”与“有挑战”。</li>
              <li>若成长带内题不足，会按与成长带的距离从近到远补齐 Top‑K。</li>
            </ul>
        </div>
        <div class="card" style="text-align:center">
             <h3>推荐难度分布</h3>
             <div class="chart-container">
                <img class="viz-img-lg" src="data:image/png;base64,{img_b64}" alt="推荐难度分布">
             </div>
             <div class="help">柱越高表示推荐题在该难度更集中，用于检查“是否符合成长带预期”。</div>
        </div>
    </div>

    <div class="card">
        <h3>📋 推荐结果</h3>
        <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                    <th width="80">ID</th>
                    <th>题目名称</th>
                    <th width="80">难度</th>
                    <th>标签</th>
                    <th width="120">推荐语言</th>
                    <th width="120">预测通过率</th>
                </tr>
              </thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
    </div>

{HTML_FOOTER}
""".encode("utf-8")
        self._send(200, body, "text/html; charset=utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
