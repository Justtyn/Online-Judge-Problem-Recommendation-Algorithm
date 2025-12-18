"""
05_make_eda_plots.py

用途
- 基于 `CleanData/` 下的清洗后数据做基础 EDA（探索性数据分析）与可视化。
- 产出分布/关系图，统一保存到 `Reports/fig/fig_*.png`，用于报告撰写与快速 sanity-check。

主要输入
- `CleanData/submissions.csv`：提交记录（含 user_id/problem_id/language/ac/attempt_no 等）
- `CleanData/problems.csv`：题目信息（含 difficulty/tags 等）
- `CleanData/students_derived.csv`：学生画像派生字段（level/perseverance 等）
- `CleanData/tags.csv`：标签词表（tag_name）
- `CleanData/languages.csv`：语言词表（name）

说明
- 强制 matplotlib 使用 Agg 后端，保证在无 GUI 的环境可运行（如服务器/CI）。
- matplotlib 缓存/配置写入仓库 `.cache/`，避免污染用户目录。
"""

import json
import os
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent

#
# 运行环境准备：把缓存/配置写入仓库内，避免污染用户目录 & 便于 CI/容器
#
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBMISSIONS = ROOT / "CleanData/submissions.csv"
PROBLEMS = ROOT / "CleanData/problems.csv"
STUDENTS_DERIVED = ROOT / "CleanData/students_derived.csv"
TAGS = ROOT / "CleanData/tags.csv"
LANGS = ROOT / "CleanData/languages.csv"

# 图表统一收敛到 Reports/fig/
OUT_DIR = ROOT / "Reports/fig"


def setup_cn_font() -> None:
    """配置 matplotlib 中文字体回退，保证图表标题/标签可显示中文。"""
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


def parse_json_list(x: object) -> list[str]:
    """
    解析“列表字段”（常见于 tags），尽量兼容多种上游格式。

    支持输入形式
    - NaN/空字符串：返回 []
    - Python list：直接转成 str 列表
    - JSON 字符串（如 '["dp","graph"]'）：优先 json.loads
    - 其它分隔字符串：按 `,`/`;`/`|` 等分割
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


def save_fig(path: Path) -> None:
    """保存当前 matplotlib 图像到指定路径，并关闭 figure（避免内存/句柄累积）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def read_csv_required(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    """
    读取 CSV 并检查必需列是否存在；缺失时给出更明确的错误信息。

    EDA 的定位是“快速发现数据异常”，因此这里对输入做最基本的校验。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列：{missing}")
    return df


def to_int_or_raise(s: pd.Series, *, field_name: str) -> pd.Series:
    """
    将字段转换为 int；若出现无法解析的值，则抛出带字段名的异常。

    说明：`CleanData/` 理论上已完成清洗；若这里报错，通常意味着上游数据存在异常或漏清洗。
    """
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.isna().any():
        bad_cnt = int(s_num.isna().sum())
        raise ValueError(f"{field_name} 存在无法解析为整数的值（NaN after coerce），数量={bad_cnt}")
    return s_num.astype(int)


def main() -> int:
    """
    生成 EDA 图表并输出到 `Reports/fig/`。

    图表大体分两类：
    - “画像/总体分布”：level/perseverance、语言分布、标签分布、用户活跃度
    - “因素 vs 通过率（AC率）”：难度/尝试次数/语言/标签 与 AC 率
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) 读取数据（CleanData 应来自前置清洗流程）----
    subs = read_csv_required(
        SUBMISSIONS,
        required_cols=["user_id", "problem_id", "language", "attempt_no", "ac"],
    )
    subs["user_id"] = to_int_or_raise(subs["user_id"], field_name="submissions.user_id")
    subs["problem_id"] = to_int_or_raise(subs["problem_id"], field_name="submissions.problem_id")
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    problems = read_csv_required(PROBLEMS, required_cols=["problem_id", "difficulty", "tags"])
    problems["problem_id"] = to_int_or_raise(problems["problem_id"], field_name="problems.problem_id")
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    # difficulty 允许缺失：用中位数回填（更稳健，且避免 drop 掉题目）
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    tags = read_csv_required(TAGS, required_cols=["tag_name"])
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)

    langs = read_csv_required(LANGS, required_cols=["name"])
    lang_vocab = langs["name"].astype(str).tolist()
    lang_set = set(lang_vocab)

    students = read_csv_required(STUDENTS_DERIVED, required_cols=["level", "perseverance"])
    students["level"] = pd.to_numeric(students["level"], errors="coerce").fillna(0.0)
    students["perseverance"] = pd.to_numeric(students["perseverance"], errors="coerce").fillna(0.0)

    # ---- 2) 画像/总体分布类图表 ----
    plt.figure()
    plt.hist(students["level"], bins=20, edgecolor="black")
    plt.title("用户能力（level）分布")
    plt.xlabel("level（0-1）")
    plt.ylabel("人数")
    save_fig(OUT_DIR / "fig_level_hist.png")

    plt.figure()
    plt.hist(students["perseverance"], bins=20, edgecolor="black")
    plt.title("用户坚持度（perseverance）分布")
    plt.xlabel("perseverance（0-1）")
    plt.ylabel("人数")
    save_fig(OUT_DIR / "fig_perseverance_hist.png")

    lang_counts = subs.groupby("language").size().sort_values(ascending=False)
    if len(lang_counts):
        # 让图表顺序与词表一致，便于多次运行对比（避免排序波动）
        lang_counts = lang_counts.reindex([l for l in lang_vocab if l in lang_counts.index], fill_value=0)
        plt.figure(figsize=(8, 4))
        plt.bar(lang_counts.index.astype(str), lang_counts.values)
        plt.title("语言总体分布（按提交）")
        plt.xlabel("语言")
        plt.ylabel("提交次数")
        plt.xticks(rotation=30, ha="right")
        save_fig(OUT_DIR / "fig_lang_dist.png")

    tag_rows = []
    for lst in problems["tags_list"].tolist():
        if not isinstance(lst, list):
            continue
        for t in lst:
            ts = str(t)
            if ts in tag_set:
                tag_rows.append(ts)
    tag_counts = pd.Series(tag_rows).value_counts()
    tag_counts = tag_counts.reindex(tag_vocab, fill_value=0)
    plt.figure(figsize=(10, 4))
    plt.bar(tag_counts.index.astype(str), tag_counts.values)
    plt.title("标签总体分布（按题目标签出现次数）")
    plt.xlabel("标签")
    plt.ylabel("出现次数")
    plt.xticks(rotation=45, ha="right")
    save_fig(OUT_DIR / "fig_tag_dist.png")

    user_sub_cnt = subs.groupby("user_id").size().rename("submissions")
    user_prob_cnt = subs.groupby("user_id")["problem_id"].nunique().rename("unique_problems")
    act = pd.concat([user_sub_cnt, user_prob_cnt], axis=1).fillna(0).astype(int)
    plt.figure(figsize=(9, 4))
    plt.hist(act["submissions"], bins=40, edgecolor="black")
    plt.title("用户活跃度：人均提交次数分布")
    plt.xlabel("提交次数")
    plt.ylabel("人数")
    save_fig(OUT_DIR / "fig_user_activity.png")

    # ---- 3) 因素 vs 通过率（AC率）类图表 ----
    sp = subs.merge(
        problems[["problem_id", "difficulty_filled", "tags_list"]],
        on="problem_id",
        how="left",
    )

    diff_stats = sp.groupby("difficulty_filled")["ac"].mean().reindex(range(1, 11), fill_value=np.nan)
    plt.figure()
    plt.plot(diff_stats.index, diff_stats.values, marker="o")
    plt.title("难度 vs 通过率（AC率）")
    plt.xlabel("难度（1-10）")
    plt.ylabel("通过率")
    plt.ylim(0, 1)
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    save_fig(OUT_DIR / "fig_difficulty_vs_ac.png")

    att = sp.copy()
    # attempt_no 可能有长尾：截断到 10 方便可视化（10 表示 “>=10”）
    att["attempt_bucket"] = att["attempt_no"].clip(upper=10)
    att_stats = att.groupby("attempt_bucket")["ac"].mean().reindex(range(1, 11), fill_value=np.nan)
    plt.figure()
    plt.plot(att_stats.index, att_stats.values, marker="o")
    plt.title("尝试次数 vs 通过率（attempt_no，截断到10）")
    plt.xlabel("attempt_no（1-10；10代表≥10）")
    plt.ylabel("通过率")
    plt.ylim(0, 1)
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    save_fig(OUT_DIR / "fig_attemptno_vs_ac.png")

    lang_ac = sp[sp["language"].isin(lang_set)].groupby("language")["ac"].mean()
    lang_ac = lang_ac.reindex([l for l in lang_vocab if l in lang_ac.index], fill_value=np.nan)
    plt.figure(figsize=(8, 4))
    plt.bar(lang_ac.index.astype(str), lang_ac.values)
    plt.title("各语言平均通过率（AC率）")
    plt.xlabel("语言")
    plt.ylabel("通过率")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    save_fig(OUT_DIR / "fig_lang_acrate.png")

    # 多标签：一个 submission 可能对应多个 tags；将 tags “展开”为多行再统计标签 AC 率
    t_rows = []
    for uid, ac, lst in sp[["user_id", "ac", "tags_list"]].itertuples(index=False):
        if not isinstance(lst, list):
            continue
        for t in lst:
            ts = str(t)
            if ts in tag_set:
                t_rows.append((ts, int(ac)))
    if t_rows:
        tdf = pd.DataFrame(t_rows, columns=["tag", "ac"])
        tag_ac = tdf.groupby("tag")["ac"].mean().reindex(tag_vocab, fill_value=np.nan)
        plt.figure(figsize=(10, 4))
        plt.bar(tag_ac.index.astype(str), tag_ac.values)
        plt.title("各标签平均通过率（AC率）")
        plt.xlabel("标签")
        plt.ylabel("通过率")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        save_fig(OUT_DIR / "fig_tag_acrate.png")

    print("已生成图表到", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
