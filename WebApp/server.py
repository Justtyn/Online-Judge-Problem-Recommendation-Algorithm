import argparse
import base64
import html
import io
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
    vals = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
    s = sum(vals.values())
    if s <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / s for k, v in vals.items()}


class Recommender:
    def __init__(self) -> None:
        self.train_samples = pd.read_csv(FEATUREDATA_DIR / "train_samples.csv")
        for col in ("submission_id", "user_id", "problem_id"):
            if col in self.train_samples.columns:
                self.train_samples[col] = pd.to_numeric(self.train_samples[col], errors="coerce")

        self.feature_cols = [
            c
            for c in self.train_samples.columns
            if c not in {"ac", "submission_id", "user_id", "problem_id"}
        ]
        self.col_to_idx = {c: i for i, c in enumerate(self.feature_cols)}

        required = {"attempt_no", "difficulty_filled", "level", "perseverance", "lang_match", "tag_match"}
        missing = sorted(required - set(self.feature_cols))
        if missing:
            raise RuntimeError(
                f"train_samples.csv 缺少列 {missing}；请先运行 `python 04_build_features.py` 重新生成。"
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

        X = self.train_samples[self.feature_cols].to_numpy(dtype=np.float32)
        y = self.train_samples["ac"].astype(int).to_numpy()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, random_state=42)),
            ]
        )
        self.model.fit(X, y)

        self.problems = pd.read_csv(CLEANDATA_DIR / "problems.csv")
        self.problems["problem_id"] = pd.to_numeric(self.problems["problem_id"], errors="coerce").astype(int)
        self.problems["difficulty"] = pd.to_numeric(self.problems["difficulty"], errors="coerce")
        diff_median = int(np.nanmedian(self.problems["difficulty"])) if self.problems["difficulty"].notna().any() else 5
        self.problems["difficulty_filled"] = self.problems["difficulty"].fillna(diff_median).astype(int)

        tags_df = pd.read_csv(CLEANDATA_DIR / "tags.csv")
        self.tag_vocab = tags_df["tag_name"].astype(str).tolist()
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
        self.problem_ids = self.problems["problem_id"].to_numpy(dtype=np.int32)
        self.problem_diff = self.problems["difficulty_filled"].to_numpy(dtype=np.int32)
        self.problem_tags_mh = np.zeros((len(self.problems), len(self.tag_names)), dtype=np.uint8)
        for i, tags_list in enumerate(self.problems["tags_list"].tolist()):
            if not isinstance(tags_list, list):
                continue
            for t in tags_list[:2]:
                j = self.tag_to_j.get(str(t))
                if j is not None:
                    self.problem_tags_mh[i, j] = 1
        self.problem_tag_counts = self.problem_tags_mh.sum(axis=1).astype(np.float32)

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

FIG_INFO: dict[str, dict[str, str]] = {
    "fig_level_hist.png": {
        "title": "用户能力（level）分布",
        "desc": "用于检查能力分层是否合理；一般应有差异与长尾，而不是全部集中在某个区间。",
    },
    "fig_perseverance_hist.png": {
        "title": "用户坚持度（perseverance）分布",
        "desc": "用于检查重试/坚持差异；不应全部接近 0 或 1。",
    },
    "fig_lang_dist.png": {
        "title": "语言总体分布（按提交）",
        "desc": "检查语言占比是否符合常识；也可支持“语言特征有效性”的论证。",
    },
    "fig_tag_dist.png": {
        "title": "标签总体分布（题型占比）",
        "desc": "检查 12 类题型分布是否合理，避免极端失衡影响训练。",
    },
    "fig_user_activity.png": {
        "title": "用户活跃度分布（提交次数）",
        "desc": "观察长尾：少数高活跃用户 + 大量低活跃用户通常更符合真实平台。",
    },
    "fig_difficulty_vs_ac.png": {
        "title": "难度 vs 通过率（AC率）",
        "desc": "关键合理性校验：难度越高，通过率应整体下降（负相关）。",
    },
    "fig_attemptno_vs_ac.png": {
        "title": "尝试次数 vs 通过率（attempt_no）",
        "desc": "观察多次尝试是否有“学习/纠错”效应；趋势应可解释。",
    },
    "fig_tag_acrate.png": {
        "title": "各标签平均通过率（AC率）",
        "desc": "对比不同题型的难度差异，证明“标签特征”有信息量。",
    },
    "fig_lang_acrate.png": {
        "title": "各语言平均通过率（AC率）",
        "desc": "对比不同语言的通过率差异，检验“语言特征”是否存在相关性。",
    },
    "fig_model_f1_compare.png": {
        "title": "模型 F1 对比（时间切分）",
        "desc": "主模型与对比模型的整体效果对比，用于“实验结果与分析”。",
    },
    "fig_cm_logreg.png": {"title": "混淆矩阵：逻辑回归", "desc": "查看 TP/FP/FN/TN 结构，结合 Precision/Recall 解释误差。"},
    "fig_cm_tree.png": {"title": "混淆矩阵：决策树", "desc": "对比不同模型的错误类型，辅助分析过拟合/欠拟合。"},
    "fig_cm_svm_or_knn.png": {"title": "混淆矩阵：SVM/KNN（对比）", "desc": "对比模型误判结构，判断是否需要更强特征或调参。"},
    "fig_hitk_curve.png": {
        "title": "Hit@K 曲线（命中=测试窗口内是否AC）",
        "desc": "推荐评估主图：K 增大命中率通常上升；曲线形状反映边际收益。",
    },
    "fig_reco_difficulty_hist.png": {
        "title": "推荐题难度分布（用户案例）",
        "desc": "验证“成长型推荐”：推荐题难度应集中在适度区间，而非全易/全难。",
    },
    "fig_reco_coverage.png": {
        "title": "推荐集中度与覆盖率",
        "desc": "检查是否只推荐少数热门题；覆盖率越高，推荐越不易同质化。",
    },
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
        ["fig_model_f1_compare.png", "fig_cm_logreg.png", "fig_cm_tree.png", "fig_cm_svm_or_knn.png"],
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
        p = urlparse(self.path)
        if p.path == "/static/style.css":
            self._send(200, STYLE_CSS.encode("utf-8"), "text/css; charset=utf-8")
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
                info = FIG_INFO.get(fn, {"title": fn, "desc": "（未登记说明）"})
                return (
                    f'<div class="card">'
                    f'<div class="muted" style="font-family:monospace">{html.escape(fn)}</div>'
                    f'<h3 style="margin:8px 0 6px">{html.escape(info["title"])}</h3>'
                    f'<a href="/reports/{html.escape(fn)}" target="_blank">'
                    f'<img src="/reports/{html.escape(fn)}" alt="{html.escape(fn)}" loading="lazy"></a>'
                    f'<div class="muted">{html.escape(info["desc"])}</div>'
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
            global RECO
            if RECO is None:
                RECO = Recommender()
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
            RECO = Recommender()

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
