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

SUBMISSIONS = "CleanData/submissions.csv"
PROBLEMS = "CleanData/problems.csv"
STUDENTS_DERIVED = "CleanData/students_derived.csv"
TAGS = "CleanData/tags.csv"
LANGS = "CleanData/languages.csv"

OUT_DIR = "Reports"


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


def save_fig(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    subs = pd.read_csv(SUBMISSIONS)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    problems = pd.read_csv(PROBLEMS)
    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    tags = pd.read_csv(TAGS)
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)

    langs = pd.read_csv(LANGS)
    lang_vocab = langs["name"].astype(str).tolist()
    lang_set = set(lang_vocab)

    students = pd.read_csv(STUDENTS_DERIVED)
    students["level"] = pd.to_numeric(students["level"], errors="coerce").fillna(0.0)
    students["perseverance"] = pd.to_numeric(students["perseverance"], errors="coerce").fillna(0.0)

    # ---- Step 3 plots ----
    plt.figure()
    plt.hist(students["level"], bins=20, edgecolor="black")
    plt.title("用户能力（level）分布")
    plt.xlabel("level（0-1）")
    plt.ylabel("人数")
    save_fig(f"{OUT_DIR}/fig_level_hist.png")

    plt.figure()
    plt.hist(students["perseverance"], bins=20, edgecolor="black")
    plt.title("用户坚持度（perseverance）分布")
    plt.xlabel("perseverance（0-1）")
    plt.ylabel("人数")
    save_fig(f"{OUT_DIR}/fig_perseverance_hist.png")

    lang_counts = subs.groupby("language").size().sort_values(ascending=False)
    if len(lang_counts):
        lang_counts = lang_counts.reindex([l for l in lang_vocab if l in lang_counts.index], fill_value=0)
        plt.figure(figsize=(8, 4))
        plt.bar(lang_counts.index.astype(str), lang_counts.values)
        plt.title("语言总体分布（按提交）")
        plt.xlabel("语言")
        plt.ylabel("提交次数")
        plt.xticks(rotation=30, ha="right")
        save_fig(f"{OUT_DIR}/fig_lang_dist.png")

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
    save_fig(f"{OUT_DIR}/fig_tag_dist.png")

    user_sub_cnt = subs.groupby("user_id").size().rename("submissions")
    user_prob_cnt = subs.groupby("user_id")["problem_id"].nunique().rename("unique_problems")
    act = pd.concat([user_sub_cnt, user_prob_cnt], axis=1).fillna(0).astype(int)
    plt.figure(figsize=(9, 4))
    plt.hist(act["submissions"], bins=40, edgecolor="black")
    plt.title("用户活跃度：人均提交次数分布")
    plt.xlabel("提交次数")
    plt.ylabel("人数")
    save_fig(f"{OUT_DIR}/fig_user_activity.png")

    # ---- Step 4 plots ----
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
    save_fig(f"{OUT_DIR}/fig_difficulty_vs_ac.png")

    att = sp.copy()
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
    save_fig(f"{OUT_DIR}/fig_attemptno_vs_ac.png")

    lang_ac = sp[sp["language"].isin(lang_set)].groupby("language")["ac"].mean()
    lang_ac = lang_ac.reindex([l for l in lang_vocab if l in lang_ac.index], fill_value=np.nan)
    plt.figure(figsize=(8, 4))
    plt.bar(lang_ac.index.astype(str), lang_ac.values)
    plt.title("各语言平均通过率（AC率）")
    plt.xlabel("语言")
    plt.ylabel("通过率")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    save_fig(f"{OUT_DIR}/fig_lang_acrate.png")

    # explode tags per submission (multi-label)
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
        save_fig(f"{OUT_DIR}/fig_tag_acrate.png")

    print("已生成图表到", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
