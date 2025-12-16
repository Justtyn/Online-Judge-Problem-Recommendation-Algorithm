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
        level, perseverance, lang_pref, tag_pref, solved, attempt_next_map = self._profile_from_history(user_df, cutoff_id)

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

        idx_band = idx_all[in_band[idx_all]]
        idx_other = idx_all[~in_band[idx_all]]
        idx_band_sorted = idx_band[np.argsort(score[idx_band])[::-1]]
        idx_other_sorted = idx_other[np.argsort(score[idx_other])[::-1]]
        picks = np.concatenate([idx_band_sorted, idx_other_sorted], axis=0)[:k]

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
        df = df.sort_values("score", ascending=False)

        band = df[(df["score"] >= min_p) & (df["score"] <= max_p)]
        if len(band) >= k:
            out = band.head(k).copy()
        else:
            out = pd.concat([band, df[~df.index.isin(band.index)]], axis=0).head(k).copy()

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

# 优化的 CSS，现代化设计
STYLE_CSS = """
:root {
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --radius: 12px;
}

body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
    background-color: var(--bg-color);
    color: var(--text-main);
    margin: 0;
    padding: 20px;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 15px;
}

h1, h2, h3 {
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #0f172a;
    margin-bottom: 1rem;
}

h1 { font-size: 1.875rem; margin-top: 2rem; }
h2 { font-size: 1.5rem; margin-top: 1.5rem; }

a {
    color: var(--primary-color);
    text-decoration: none;
    transition: color 0.2s;
}

a:hover {
    color: var(--primary-hover);
    text-decoration: underline;
}

/* Card Styling */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 24px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 24px;
}

.card img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    display: block;
    margin: 10px auto;
}

/* Form Elements */
label {
    display: block;
    font-weight: 500;
    margin: 0 0 8px;
    color: #334155;
    font-size: 0.95rem;
}

input[type="text"], textarea, select {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 14px;
    background: #fff;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-sizing: border-box;
}

input[type="text"]:focus, textarea:focus, select:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

/* Custom Range Slider */
input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    background: transparent;
    margin: 10px 0;
}
input[type=range]:focus { outline: none; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 18px;
    width: 18px;
    border-radius: 50%;
    background: var(--primary-color);
    cursor: pointer;
    margin-top: -7px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 4px;
    cursor: pointer;
    background: #cbd5e1;
    border-radius: 2px;
}

/* Buttons */
button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
}

button[type="submit"] {
    background-color: var(--primary-color);
    color: white;
}
button[type="submit"]:hover {
    background-color: var(--primary-hover);
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
}

.btn-secondary {
    background-color: #fff;
    color: var(--text-main);
    border: 1px solid var(--border-color);
}
.btn-secondary:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}

.actions {
    display: flex;
    gap: 12px;
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
}

/* Table Styling */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}
th {
    background-color: #f1f5f9;
    font-weight: 600;
    text-align: left;
    padding: 12px 16px;
    color: #475569;
    border-bottom: 2px solid var(--border-color);
}
td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    vertical-align: middle;
}
tr:hover td {
    background-color: #f8fafc;
}

/* Utility */
.muted { color: var(--text-muted); font-size: 0.875rem; line-height: 1.4; margin-top: 4px; }
.row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.row output { font-family: monospace; font-weight: 600; color: var(--primary-color); width: 45px; text-align: right; }
.pill { display: inline-block; padding: 2px 8px; font-size: 12px; border-radius: 99px; background: #e2e8f0; color: #475569; font-weight: 600; }
.subgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 24px; }
.header-actions { margin-bottom: 20px; display: flex; align-items: center; gap: 15px; font-size: 0.95rem; }
details { border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; background: #fafafa; }
summary { cursor: pointer; font-weight: 500; color: var(--text-main); }
"""

FIG_INFO: dict[str, dict[str, object]] = {
    "fig_level_hist.png": {
        "title": "用户能力（level）分布",
        "summary": "用于验证“能力画像”是否合理，以及是否存在异常（全 0 / 全 1 / 过度集中）。",
        "how": [
            "横轴是 level（0~1），纵轴是人数；越靠右表示能力越强。",
            "如果大部分用户集中在很窄的区间，说明画像区分度不足；如果两端极端多，可能是归一化/公式问题。",
        ],
        "tips": [
            "结合 fig_user_activity.png 看：低活跃用户占比高时，level 可能更不稳定。",
            "若 level 几乎不随时间变化，可能存在“看未来”的时间泄漏（需要严格按历史计算画像）。",
        ],
    },
    "fig_perseverance_hist.png": {
        "title": "用户坚持度（perseverance）分布",
        "summary": "用于验证“坚持/重试”画像是否有差异，以及是否过度饱和到 0 或 1。",
        "how": [
            "横轴是 perseverance（0~1），纵轴是人数；数值越大表示平均重试/坚持更强。",
            "如果大量用户都接近 1，说明归一化尺度可能太小或分母设置不合理。",
        ],
        "tips": [
            "perseverance 通常受 attempt_no 分布影响，建议对照 fig_attemptno_vs_ac.png 解释。",
        ],
    },
    "fig_lang_dist.png": {
        "title": "语言总体分布（按提交）",
        "summary": "用于证明语言数据分布符合常识，并说明“语言特征”可能有区分信息。",
        "how": [
            "横轴是语言，纵轴是提交次数。",
            "如果某些语言几乎为 0，可能是数据缺失/语言白名单不匹配；会导致 one-hot 特征弱或无效。",
        ],
    },
    "fig_tag_dist.png": {
        "title": "标签总体分布（题型占比）",
        "summary": "用于检查题型空间是否失衡；极端失衡会让模型更偏向少数标签。",
        "how": [
            "横轴是标签，纵轴是出现次数（题目标签出现）。",
            "少数标签特别高会造成训练样本集中，推荐也容易同质化。",
        ],
    },
    "fig_user_activity.png": {
        "title": "用户活跃度分布（提交次数）",
        "summary": "用于展示平台常见长尾：少数高活跃用户 + 大量低活跃用户。",
        "how": [
            "横轴是提交次数，纵轴是人数；通常会出现明显长尾。",
            "低活跃用户多时，画像/偏好统计更不稳定，推荐更难做得很准。",
        ],
    },
    "fig_difficulty_vs_ac.png": {
        "title": "难度 vs 通过率（AC率）",
        "summary": "关键合理性校验：难度越高，AC 率应整体下降（负相关）。",
        "how": [
            "横轴是难度（1~10），纵轴是通过率（0~1）。",
            "如果曲线不降反升或剧烈抖动，可能是 difficulty 标注不可靠或数据量不足。",
        ],
    },
    "fig_attemptno_vs_ac.png": {
        "title": "尝试次数 vs 通过率（attempt_no）",
        "summary": "用于解释“多次尝试与通过率”的关系，帮助理解 attempt_no 特征的作用。",
        "how": [
            "横轴是 attempt_no（第几次尝试），纵轴是 AC 率。",
            "如果 attempt_no 越大 AC 率越高，说明有学习/纠错效应；反之可能表示越难的题需要更多尝试。",
        ],
    },
    "fig_tag_acrate.png": {
        "title": "各标签平均通过率（AC率）",
        "summary": "用于证明不同题型的难度差异，以及标签特征与 AC 的相关性。",
        "how": [
            "横轴是标签，纵轴是平均 AC 率。",
            "如果所有标签 AC 率几乎一样，说明标签区分信息较弱或统计口径有问题。",
        ],
    },
    "fig_lang_acrate.png": {
        "title": "各语言平均通过率（AC率）",
        "summary": "用于检验“语言特征”是否与通过率存在相关性（并不表示因果）。",
        "how": [
            "横轴是语言，纵轴是平均 AC 率。",
            "差异过大可能来自用户群体差异（强者偏某语言）或题目选择偏差，不建议做因果解读。",
        ],
    },
    "fig_model_f1_compare.png": {
        "title": "模型 F1 对比（时间切分）",
        "summary": "用于对比多个模型在测试窗口的整体分类质量（兼顾 Precision 与 Recall）。",
        "how": [
            "柱越高表示该模型对“是否 AC”预测更稳定。",
            "如果不同模型差距很小，通常说明特征决定了上限；可考虑特征改进或更贴近推荐目标的建模方式。",
        ],
    },
    "fig_hitk_curve.png": {
        "title": "Hit@K 曲线（多策略对比）",
        "summary": "用于对比不同推荐策略的命中率：K 越大通常越容易命中，但边际收益会下降。",
        "how": [
            "横轴是 K（推荐列表长度），纵轴是 Hit@K（测试窗口内是否命中过至少 1 道最终 AC 的题）。",
            "重点看：model vs random 是否显著更高（证明模型有效）；popular_train 的位置可作为强基线参照。",
            "growth 策略可能牺牲部分命中率以换取更适度的学习题目（需要结合难度分布图解释）。",
        ],
    },
    "fig_reco_difficulty_hist.png": {
        "title": "推荐题难度分布（用户案例）",
        "summary": "用于解释推荐列表的“难度结构”，尤其是成长带（ZPD）策略是否在推“刚好够得着”的题。",
        "how": [
            "横轴是难度（1~10），纵轴是推荐题数量。",
            "如果全部集中在最低难度，推荐缺挑战；如果全部集中在最高难度，命中与可学习性都会差。",
        ],
    },
    "fig_reco_coverage.png": {
        "title": "推荐集中度与覆盖率",
        "summary": "用于判断推荐是否过度集中在少数热门题（同质化），以及整体覆盖率。",
        "how": [
            "柱状图展示 Top20 被推荐次数最多的题；标题里有 coverage（覆盖率）。",
            "如果 Top20 柱子极高且 coverage 很低，说明推荐同质化严重；可用候选过滤/多样性重排改进。",
        ],
    },
}


def _html_ul(items: list[str]) -> str:
    if not items:
        return ""
    return "<ul style='margin:8px 0 0; padding-left:18px'>" + "".join(
        f"<li>{html.escape(x)}</li>" for x in items if str(x).strip()
    ) + "</ul>"


def get_fig_info(filename: str) -> dict[str, object]:
    if filename in FIG_INFO:
        return FIG_INFO[filename]

    # Confusion matrix family
    if filename.startswith("fig_cm_") or filename.startswith("fig_confusion_"):
        name = filename.removeprefix("fig_cm_").removeprefix("fig_confusion_").removesuffix(".png")
        title_map = {
            "logreg": "逻辑回归",
            "tree": "决策树",
            "svm_linear": "线性 SVM",
            "svm_or_knn": "SVM/KNN（对比）",
        }
        model_name = title_map.get(name, name)
        return {
            "title": f"混淆矩阵：{model_name}",
            "summary": "用于拆解模型错误类型（把 AC 当作正类），帮助解释 Precision/Recall 为什么会这样。",
            "how": [
                "矩阵对角线越高越好：左上=真负（预测未AC且确实未AC），右下=真正（预测AC且确实AC）。",
                "右上=假正（误报AC，影响 Precision）；左下=假负（漏报AC，影响 Recall）。",
            ],
            "tips": [
                "如果假负很多：模型保守，可能需要更强特征或调阈值；如果假正很多：模型过于乐观。",
            ],
        }

    # Compare figures (strict vs leaky)
    if filename.startswith("fig_compare_"):
        key = filename.removesuffix(".png").removeprefix("fig_compare_")
        title_map = {
            "hitk": "Hit@K 对比（strict vs leaky）",
            "precisionk": "Precision@K 对比（strict vs leaky）",
            "roc": "ROC 曲线对比（strict vs leaky）",
            "pr": "PR 曲线对比（strict vs leaky）",
            "calibration": "校准曲线对比（strict vs leaky）",
        }
        title = title_map.get(key, f"对比图：{key}")
        return {
            "title": title,
            "summary": "用于判断旧口径是否“失真”（leaky 看未来会抬高指标），strict 更接近真实可部署效果。",
            "how": [
                "如果 leaky 明显高于 strict：说明过去评估被时间泄漏抬高；报告应以 strict 为准。",
                "如果两者接近：说明时间泄漏影响不大，模型/特征更可信。",
            ],
            "tips": [
                "ROC/PR 侧重排序能力；校准曲线侧重概率是否可信（是否过于乐观/保守）。",
            ],
        }

    # Generic fallback based on filename patterns
    if "acrate" in filename or "vs_ac" in filename:
        return {
            "title": filename,
            "summary": "用于查看某个因素（难度/尝试次数/语言/标签）与 AC 率的关系，帮助验证特征是否有信息量。",
            "how": [
                "先看横轴是什么特征，纵轴通常是 AC 率（0~1）。",
                "趋势是否符合直觉：难度升高通过率下降；标签/语言差异应可解释。",
            ],
        }
    if "hist" in filename or "dist" in filename:
        return {
            "title": filename,
            "summary": "分布图：用于检查数据是否符合常识、是否存在异常值/极端集中。",
            "how": [
                "看横轴变量的取值范围，纵轴是数量/人数。",
                "重点关注：是否长尾、是否异常尖峰、是否出现不可能的取值。",
            ],
        }
    if "coverage" in filename:
        return {
            "title": filename,
            "summary": "用于检查推荐是否同质化（是否总推荐少数题），以及覆盖率是否足够。",
            "how": [
                "Top20 柱越高且越集中，说明推荐越同质化；覆盖率越高表示推荐更分散。",
            ],
        }

    return {
        "title": filename,
        "summary": "未登记说明：建议先看图的标题与坐标轴含义，再结合上下游图表做解释。",
        "how": [
            "若是训练相关图：结合 Models/metrics.csv 的指标理解好坏。",
            "若是推荐相关图：结合 Reports/reco_metrics.csv 的 Hit@K/Precision@K/Recall@K/NDCG@K 理解好坏。",
        ],
    }

FIG_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "A. 数据层可视化（训练前）",
        [
            "fig_level_hist.png",
            "fig_perseverance_hist.png",
            "fig_lang_dist.png",
            "fig_tag_dist.png",
            "fig_user_activity.png",
            "fig_difficulty_vs_ac.png",
            "fig_attemptno_vs_ac.png",
            "fig_tag_acrate.png",
            "fig_lang_acrate.png",
        ],
    ),
    (
        "B. 训练层可视化（训练后）",
        [
            "fig_model_f1_compare.png",
            "fig_cm_logreg.png",
            "fig_cm_tree.png",
            "fig_cm_svm_or_knn.png",
            "fig_cm_svm_linear.png",
            "fig_confusion_logreg.png",
            "fig_confusion_tree.png",
            "fig_confusion_svm_linear.png",
        ],
    ),
    ("C. 推荐评估（Top-K）", ["fig_hitk_curve.png", "fig_reco_difficulty_hist.png", "fig_reco_coverage.png"]),
]


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
        if p.path == "/static/style.css":
            self._send(200, STYLE_CSS.encode("utf-8"), "text/css; charset=utf-8")
            return

        if p.path == "/api/student":
            if RECO is None:
                try:
                    RECO = Recommender()
                except Exception as e:
                    self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                    return

            q = parse_qs(p.query or "")

            def q1(name: str, default: str = "") -> str:
                v = q.get(name, [])
                return v[0] if v else default

            try:
                user_id = int(q1("user_id", "1"))
                pct = float(q1("pct", "0.5"))
                k = int(q1("k", "10"))
                min_p = float(q1("min_p", "0.4"))
                max_p = float(q1("max_p", "0.7"))
            except Exception:
                self._send(400, "参数格式错误".encode("utf-8"), "text/plain; charset=utf-8")
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

        if p.path == "/student":
            body = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>单学生动态展示 - OJ系统</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div class="container">
    <div style="margin-bottom: 20px;">
      <a href="/" style="font-size: 14px;">&larr; 返回首页</a>
      <span style="margin: 0 10px; color:#ccc">|</span>
      <a href="/custom" style="font-size: 14px;">自定义推荐</a>
    </div>

    <h1>👤 单学生动态展示</h1>
    <div class="card">
      <div class="subgrid">
        <div>
          <label for="user_id">user_id</label>
          <input id="user_id" type="text" value="1" placeholder="例如 1">
          <div class="muted">从 <span style="font-family:monospace">CleanData/submissions.csv</span> 选择存在的 user_id。</div>
        </div>
        <div>
          <label for="pct">时间点（按该学生提交序列百分位）</label>
          <div class="row">
            <input id="pct" type="range" min="0" max="1" step="0.01" value="0.50">
            <output id="pct_out">0.50</output>
          </div>
          <div class="muted">0=最早，1=最新；用于生成 cutoff_submission_id。</div>
        </div>
        <div>
          <label for="k">Top K</label>
          <div class="row">
            <input id="k" type="range" min="1" max="50" step="1" value="10">
            <output id="k_out">10</output>
          </div>
        </div>
        <div>
          <label>ZPD 区间</label>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div>
              <label style="font-size:12px">min_p</label>
              <div class="row">
                <input id="min_p" type="range" min="0" max="1" step="0.01" value="0.40">
                <output id="min_p_out">0.40</output>
              </div>
            </div>
            <div>
              <label style="font-size:12px">max_p</label>
              <div class="row">
                <input id="max_p" type="range" min="0" max="1" step="0.01" value="0.70">
                <output id="max_p_out">0.70</output>
              </div>
            </div>
          </div>
          <div class="muted">优先挑选 P(AC) 落在区间内的题目，不足再用高分补齐。</div>
        </div>
      </div>
      <div class="actions">
        <button class="btn-secondary" id="refresh_btn" type="button">刷新</button>
        <div class="muted" id="status" style="align-self:center"></div>
      </div>
    </div>

    <div class="card">
      <h2 style="margin-top:0">📌 当前状态</h2>
      <div class="muted" id="meta"></div>
    </div>

    <div class="card">
      <h2 style="margin-top:0">1) 时间轴散点</h2>
      <img id="img_timeline" style="width:100%; max-width:1100px" alt="timeline_scatter">
    </div>

    <div class="card">
      <h2 style="margin-top:0">2) 雷达对比（历史 vs 推荐）</h2>
      <img id="img_radar" style="width:100%; max-width:1100px" alt="radar_compare">
    </div>

    <div class="card">
      <h2 style="margin-top:0">3) 难度阶梯（推荐列表）</h2>
      <img id="img_ladder" style="width:100%; max-width:1100px" alt="difficulty_ladder">
    </div>

    <div class="card" style="padding:0; overflow:hidden;">
      <div style="padding:24px 24px 0"><h2 style="margin:0">Top-K 推荐列表</h2></div>
      <div style="overflow-x:auto; padding: 16px 24px 24px;">
        <table>
          <thead>
            <tr>
              <th width="60">Rank</th>
              <th width="90">Problem</th>
              <th>Title</th>
              <th width="80">难度</th>
              <th>Tags</th>
              <th width="90">Language</th>
              <th width="90">P(AC)</th>
              <th width="90">In ZPD</th>
            </tr>
          </thead>
          <tbody id="reco_rows"></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
function bindOut(id, outId, fmt) {
  const el = document.getElementById(id);
  const out = document.getElementById(outId);
  const update = () => out.textContent = fmt(el.value);
  el.addEventListener("input", update);
  update();
}
bindOut("pct","pct_out",(v)=>Number(v).toFixed(2));
bindOut("k","k_out",(v)=>String(v));
bindOut("min_p","min_p_out",(v)=>Number(v).toFixed(2));
bindOut("max_p","max_p_out",(v)=>Number(v).toFixed(2));

function esc(s) {
  return String(s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

async function refresh() {
  const status = document.getElementById("status");
  status.textContent = "加载中...";
  const user_id = document.getElementById("user_id").value.trim() || "1";
  const pct = document.getElementById("pct").value;
  const k = document.getElementById("k").value;
  const min_p = document.getElementById("min_p").value;
  const max_p = document.getElementById("max_p").value;

  const url = `/api/student?user_id=${encodeURIComponent(user_id)}&pct=${encodeURIComponent(pct)}&k=${encodeURIComponent(k)}&min_p=${encodeURIComponent(min_p)}&max_p=${encodeURIComponent(max_p)}`;
  let data;
  try {
    const resp = await fetch(url);
    const text = await resp.text();
    if(!resp.ok) throw new Error(text);
    data = JSON.parse(text);
  } catch (e) {
    status.textContent = "失败：" + (e && e.message ? e.message : e);
    return;
  }

  const m = data.meta || {};
  document.getElementById("meta").innerHTML =
    `<div><b>user_id</b>: ${m.user_id} &nbsp; <b>cutoff</b>: ${m.cutoff_submission_id} (pct=${Number(m.cutoff_pct).toFixed(2)})</div>` +
    `<div><b>hist_submissions</b>: ${m.hist_submissions} &nbsp; <b>hist_solved</b>: ${m.hist_solved}</div>` +
    `<div><b>level</b>: ${Number(m.level).toFixed(3)} &nbsp; <b>perseverance</b>: ${Number(m.perseverance).toFixed(3)} &nbsp; <b>top_language</b>: ${esc(m.top_language)}</div>` +
    `<div><b>ZPD</b>: [${Number((m.zpd||[])[0] ?? 0).toFixed(2)}, ${Number((m.zpd||[])[1] ?? 1).toFixed(2)}]</div>`;

  const imgs = data.images || {};
  document.getElementById("img_timeline").src = "data:image/png;base64," + (imgs.timeline_scatter || "");
  document.getElementById("img_radar").src = "data:image/png;base64," + (imgs.radar_compare || "");
  document.getElementById("img_ladder").src = "data:image/png;base64," + (imgs.difficulty_ladder || "");

  const rows = data.recommendations || [];
  const tbody = document.getElementById("reco_rows");
  tbody.innerHTML = rows.map(r => {
    const p = Number(r.p_ac);
    const scoreStyle = p >= 0.7 ? "color:#166534;font-weight:700" : (p >= 0.4 ? "color:#ca8a04;font-weight:700" : "color:#b91c1c;font-weight:700");
    return `<tr>` +
      `<td>${r.rank}</td>` +
      `<td style="font-family:monospace;color:#666">#${r.problem_id}</td>` +
      `<td>${esc(r.title)}</td>` +
      `<td><span class="pill">${r.difficulty}</span></td>` +
      `<td class="muted">${esc(r.tags)}</td>` +
      `<td>${esc(r.language)}</td>` +
      `<td style="${scoreStyle}">${p.toFixed(4)}</td>` +
      `<td>${r.in_growth_band ? "1" : "0"}</td>` +
    `</tr>`;
  }).join("");

  status.textContent = "完成";
}

document.getElementById("refresh_btn").addEventListener("click", refresh);
["pct","k","min_p","max_p"].forEach(id => document.getElementById(id).addEventListener("change", refresh));
refresh();
</script>
</body>
</html>
""".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

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

        if p.path in {"/", "/index.html"}:
            figs = sorted([x.name for x in REPORTS_DIR.glob("fig_*.png")])
            figs_set = set(figs)

            def render_card(fn: str) -> str:
                info = get_fig_info(fn)
                title = str(info.get("title") or fn)
                summary = str(info.get("summary") or "")
                how = info.get("how") if isinstance(info.get("how"), list) else []
                tips = info.get("tips") if isinstance(info.get("tips"), list) else []
                details = ""
                blocks: list[str] = []
                if how:
                    blocks.append("<div class='muted' style='margin-top:6px'>如何理解：</div>" + _html_ul([str(x) for x in how]))
                if tips:
                    blocks.append("<div class='muted' style='margin-top:8px'>常见解读/提示：</div>" + _html_ul([str(x) for x in tips]))
                if blocks:
                    details = (
                        "<details style='margin-top:10px'>"
                        "<summary>展开：这张图怎么看</summary>"
                        + "".join(blocks)
                        + "</details>"
                    )
                return (
                    f'<div class="card">'
                    f'<div class="muted" style="font-family:monospace">{html.escape(fn)}</div>'
                    f'<h3 style="margin:8px 0 6px">{html.escape(title)}</h3>'
                    f'<a href="/reports/{html.escape(fn)}" target="_blank">'
                    f'<img src="/reports/{html.escape(fn)}" alt="{html.escape(fn)}" loading="lazy"></a>'
                    f'<div class="muted">{html.escape(summary)}</div>'
                    f"{details}"
                    f"</div>"
                )

            used: set[str] = set()
            section_blocks: list[str] = []
            for sec_title, order in FIG_SECTIONS:
                present = [fn for fn in order if fn in figs_set]
                used.update(present)
                if not present:
                    continue
                section_blocks.append(
                    f"<section style='margin-top:24px'>"
                    f"<h2>{html.escape(sec_title)}</h2>"
                    f"<div class='grid'>{''.join(render_card(fn) for fn in present)}</div>"
                    f"</section>"
                )

            others = [fn for fn in figs if fn not in used]
            if others:
                section_blocks.append(
                    f"<section style='margin-top:24px'>"
                    f"<h2>其他图表</h2>"
                    f"<div class='grid'>{''.join(render_card(fn) for fn in others)}</div>"
                    f"</section>"
                )
            body = f"""
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OJ 可视化与推荐系统</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header style="border-bottom: 1px solid #e2e8f0; margin-bottom: 30px; padding-bottom: 20px;">
            <h1>OJ 数据可视化与题目推荐</h1>
            <div class="header-actions">
                <a href="/custom" style="display:inline-flex; align-items:center; background:var(--primary-color); color:white; padding:10px 20px; border-radius:8px; text-decoration:none;">
                    <span>👉 进入个性化题目推荐</span>
                </a>
                <a href="/student" style="display:inline-flex; align-items:center; background:#fff; color:var(--primary-color); padding:10px 20px; border-radius:8px; text-decoration:none; border:1px solid var(--border-color);">
                    <span>📈 单学生动态展示</span>
                </a>
                <span class="muted">基于机器学习模型的智能推荐系统</span>
            </div>
        </header>

        <main>
            <div class="card">
                <h2 style="margin-top:0">📊 图表说明</h2>
                <p class="muted">A 类用于证明“数据分布合理/符合常识”；B 类用于展示模型效果与误差类型；C 类用于推荐评估（Hit@K、覆盖率、案例）。</p>
            </div>
            {''.join(section_blocks) if figs else '<div class="card muted">暂无图表，请先运行分析脚本。</div>'}
        </main>
    </div>
</body>
</html>
""".encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            return

        if p.path == "/custom":
            if RECO is None:
                try:
                    RECO = Recommender()
                except Exception as e:
                    self._send(500, f"WebApp 初始化失败：{e}".encode("utf-8"), "text/plain; charset=utf-8")
                    return
            tag_opts = "".join(
                f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in RECO.tag_names
            )
            lang_opts = "".join(
                f'<option value="{html.escape(l)}">{html.escape(l)}</option>' for l in RECO.lang_names
            )
            body = f"""
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>自定义学生推荐 - OJ系统</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div style="margin-bottom: 20px;">
            <a href="/" style="font-size: 14px;">&larr; 返回首页</a>
            <span style="margin: 0 10px; color:#ccc">|</span>
            <a href="/student" style="font-size: 14px;">单学生动态展示</a>
        </div>

        <h1>🎯 自定义参数推荐</h1>
        <p class="muted" style="margin-bottom: 30px;">调整以下参数来模拟不同的学生画像，系统将推荐最合适的题目。</p>

        <form method="post" action="/custom">
            <div class="card">
                <h3>基础能力画像</h3>
                <div class="subgrid">
                    <div>
                        <label for="level">能力水平 (Level)</label>
                        <div class="row">
                            <input id="level" name="level" type="range" min="0" max="1" step="0.01" value="0.50">
                            <output id="level_out">0.50</output>
                        </div>
                        <div class="muted">
                            <span class="pill">0.0</span> 新手 
                            <span style="float:right"><span class="pill">1.0</span> 大神</span>
                        </div>
                    </div>
                    <div>
                        <label for="perseverance">坚持度 (Perseverance)</label>
                        <div class="row">
                            <input id="perseverance" name="perseverance" type="range" min="0" max="1" step="0.01" value="0.60">
                            <output id="perseverance_out">0.60</output>
                        </div>
                        <div class="muted">
                            <span class="pill">低</span> 易放弃 
                            <span style="float:right"><span class="pill">高</span> 坚韧</span>
                        </div>
                    </div>
                    <div>
                        <label for="attempt_no">当前尝试次数</label>
                        <div class="row">
                            <input id="attempt_no" name="attempt_no" type="range" min="1" max="10" step="1" value="1">
                            <output id="attempt_no_out">1</output>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>推荐偏好设置</h3>
                <div class="subgrid">
                    <div>
                        <label>推荐数量 (Top K)</label>
                        <div class="row">
                            <input id="k" name="k" type="range" min="1" max="50" step="1" value="10">
                            <output id="k_out">10</output>
                        </div>
                    </div>
                    <div>
                        <label>成功率区间 (Zone of Proximal Development)</label>
                        <div class="muted" style="margin-bottom:10px;">优先推荐预测 AC 概率在此区间的题目</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <label style="font-size:12px;">Min Probability</label>
                                <div class="row">
                                    <input id="min_p" name="min_p" type="range" min="0" max="1" step="0.01" value="0.40">
                                    <output id="min_p_out">0.40</output>
                                </div>
                            </div>
                            <div>
                                <label style="font-size:12px;">Max Probability</label>
                                <div class="row">
                                    <input id="max_p" name="max_p" type="range" min="0" max="1" step="0.01" value="0.70">
                                    <output id="max_p_out">0.70</output>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="subgrid">
                <div class="card">
                    <h3>💻 编程语言偏好</h3>
                    <div style="margin-bottom:15px;">
                        <label>主用语言</label>
                        <select id="lang_top" name="lang_top">{lang_opts}</select>
                    </div>
                    <div>
                        <label>偏好强度 (权重)</label>
                        <div class="row">
                            <input id="lang_strength" name="lang_strength" type="range" min="0.50" max="0.95" step="0.01" value="0.70">
                            <output id="lang_strength_out">0.70</output>
                        </div>
                    </div>
                    <details style="margin-top:15px">
                        <summary>高级：自定义 JSON 分布</summary>
                        <div style="margin-top:10px">
                            <textarea id="lang_json" name="lang_json" rows="3" placeholder='{{"Python":0.6,"C++":0.2,"JAVA":0.2}}'></textarea>
                            <div class="muted">填写后将覆盖上方设置。</div>
                        </div>
                    </details>
                </div>

                <div class="card">
                    <h3>🏷️ 算法标签偏好</h3>
                    <div style="margin-bottom:15px;">
                        <label>感兴趣的标签 (按住 Ctrl 多选)</label>
                        <select id="tag_selected" name="tag_selected" multiple size="5" style="height: 120px;">{tag_opts}</select>
                    </div>
                    <div>
                        <label>偏好强度 (总权重)</label>
                        <div class="row">
                            <input id="tag_strength" name="tag_strength" type="range" min="0.50" max="0.95" step="0.01" value="0.70">
                            <output id="tag_strength_out">0.70</output>
                        </div>
                    </div>
                    <details style="margin-top:15px">
                        <summary>高级：自定义 JSON 分布</summary>
                        <div style="margin-top:10px">
                            <textarea id="tag_json" name="tag_json" rows="3" placeholder='{{"dp":0.4,"graph":0.3,"tree":0.3}}'></textarea>
                            <div class="muted">填写后将覆盖上方设置。</div>
                        </div>
                    </details>
                </div>
            </div>

            <div class="card">
                <h3>高级策略</h3>
                <div>
                    <label>语言评分融合模式</label>
                    <select name="mode">
                        <option value="expected" selected>期望分数模式 (推荐：按概率加权)</option>
                        <option value="best">最大分数模式 (激进：取最擅长语言的分数)</option>
                    </select>
                    <div class="muted">注：依赖 `lang_match` 和 `tag_match` 特征训练的模型。</div>
                </div>

                <div class="actions">
                    <button type="submit">✨ 生成推荐列表</button>
                    <button type="button" class="btn-secondary" id="btn_random">🎲 随机生成参数</button>
                    <button type="button" class="btn-secondary" id="btn_reset">🔄 重置</button>
                </div>
            </div>
        </form>
    </div>

<script>
function bindRange(id) {{
  const el = document.getElementById(id);
  const out = document.getElementById(id + "_out");
  if(!el || !out) return;
  const sync = () => out.textContent = Number(el.value).toFixed(el.step && el.step.includes(".") ? 2 : 0);
  el.addEventListener("input", sync);
  sync();
}}

// Initialize all sliders
["level","perseverance","attempt_no","k","min_p","max_p","lang_strength","tag_strength"].forEach(bindRange);

function clampMinMax() {{
  const minEl = document.getElementById("min_p");
  const maxEl = document.getElementById("max_p");
  if (Number(minEl.value) > Number(maxEl.value)) {{
    // Swap values if min > max
    const tmp = minEl.value;
    minEl.value = maxEl.value;
    maxEl.value = tmp;
    // Update displays manually since input event didn't fire
    document.getElementById("min_p_out").textContent = Number(minEl.value).toFixed(2);
    document.getElementById("max_p_out").textContent = Number(maxEl.value).toFixed(2);
  }}
}}
document.getElementById("min_p").addEventListener("change", clampMinMax);
document.getElementById("max_p").addEventListener("change", clampMinMax);

document.getElementById("btn_reset").addEventListener("click", () => {{
  document.getElementById("level").value = "0.50";
  document.getElementById("perseverance").value = "0.60";
  document.getElementById("attempt_no").value = "1";
  document.getElementById("k").value = "10";
  document.getElementById("min_p").value = "0.40";
  document.getElementById("max_p").value = "0.70";
  document.getElementById("lang_strength").value = "0.70";
  document.getElementById("tag_strength").value = "0.70";
  document.getElementById("lang_json").value = "";
  document.getElementById("tag_json").value = "";

  // Reset Selects
  document.getElementById("lang_top").selectedIndex = 0;
  const tagSel = document.getElementById("tag_selected");
  for(let i=0; i<tagSel.options.length; i++) tagSel.options[i].selected = false;

  // Trigger updates for sliders
  ["level","perseverance","attempt_no","k","min_p","max_p","lang_strength","tag_strength"].forEach((id)=>{{
    const el = document.getElementById(id);
    if(el) el.dispatchEvent(new Event("input"));
  }});
}});

document.getElementById("btn_random").addEventListener("click", () => {{
  const r = (a,b)=> (Number(a) + Math.random()*(b-a)).toFixed(2);
  document.getElementById("level").value = r(0.05, 0.95);
  document.getElementById("perseverance").value = r(0.05, 0.95);
  document.getElementById("attempt_no").value = String(1 + Math.floor(Math.random()*4));
  document.getElementById("k").value = String(5 + Math.floor(Math.random()*11));
  const minp = Number(r(0.25, 0.60));
  const maxp = Number(r(minp, 0.85));
  document.getElementById("min_p").value = minp.toFixed(2);
  document.getElementById("max_p").value = maxp.toFixed(2);
  document.getElementById("lang_strength").value = r(0.55, 0.90);
  document.getElementById("tag_strength").value = r(0.55, 0.90);
  document.getElementById("lang_json").value = "";
  document.getElementById("tag_json").value = "";

  // Trigger updates
  ["level","perseverance","attempt_no","k","min_p","max_p","lang_strength","tag_strength"].forEach((id)=>{{
    const el = document.getElementById(id);
    if(el) el.dispatchEvent(new Event("input"));
  }});
}});
</script>
</body>
</html>
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
        for _, r in out.iterrows():
            score_val = float(r['score'])
            # 简单的分数颜色标记
            score_style = "color:#166534; font-weight:bold;" if score_val > 0.6 else "color:#ca8a04;"

            rows.append(
                "<tr>"
                f"<td style='font-family:monospace; color:#666;'>#{int(r['problem_id'])}</td>"
                f"<td style='font-weight:500'>{html.escape(str(r.get('title') or ''))}</td>"
                f"<td><span class='pill'>{int(r['difficulty_filled'])}</span></td>"
                f"<td class='muted'>{html.escape(str(r.get('tags') or ''))}</td>"
                f"<td>{html.escape(str(r.get('language') or ''))}</td>"
                f"<td style='{score_style}'>{score_val:.4f}</td>"
                "</tr>"
            )

        body = f"""
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>推荐结果 - OJ系统</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div style="margin-bottom: 20px;">
            <a href="/custom" style="font-size: 14px;">&larr; 返回修改参数</a>
            <span style="margin: 0 10px; color:#ccc">|</span>
            <a href="/" style="font-size: 14px;">首页</a>
        </div>

        <h1>🚀 推荐题目列表</h1>

        <div class="card" style="display:flex; flex-wrap:wrap; gap:20px; align-items:flex-start;">
            <div style="flex:1; min-width:300px;">
                <h3>参数概览</h3>
                <div class="muted" style="line-height:1.8">
                    <div><b>Level:</b> {level:.3f} &nbsp; <b>Perseverance:</b> {perseverance:.3f}</div>
                    <div><b>Attempt No:</b> {attempt_no} &nbsp; <b>Top K:</b> {k}</div>
                    <div><b>ZPD Range:</b> [{min_p:.2f}, {max_p:.2f}]</div>
                    <div><b>Mode:</b> {html.escape(mode)}</div>
                </div>
            </div>
            <div style="flex:1; min-width:300px; text-align:center;">
                 <img src="data:image/png;base64,{img_b64}" alt="difficulty_hist" style="max-height:200px; border:1px solid #eee;">
            </div>
        </div>

        <div class="card" style="padding:0; overflow:hidden;">
            <div style="overflow-x:auto;">
                <table>
                  <thead>
                    <tr>
                        <th width="80">ID</th>
                        <th>题目名称</th>
                        <th width="80">难度</th>
                        <th>标签</th>
                        <th width="100">推荐语言</th>
                        <th width="100">AC 概率</th>
                    </tr>
                  </thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
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
