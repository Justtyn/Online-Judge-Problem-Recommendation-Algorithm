"""
01_derive_students.py

从 `CleanData/` 下的多张表派生“学生画像”（用户侧特征），并输出到
`CleanData/students_derived.csv`。

输入（默认路径可用 CLI 覆盖）：
- `CleanData/problems.csv`：题目元信息（含 difficulty/tags）
- `CleanData/submissions.csv`：提交记录（含 user_id/problem_id/language/ac）
- `CleanData/tags.csv`：标签词表（tag_name）
- `CleanData/languages.csv`：语言词表（name）

输出字段：
- `user_id`：用户 ID
- `level`：0~1，基于“已解题(AC) × 题目难度”的加权完成度
- `perseverance`：0~1，基于“平均每题尝试次数”的对数归一化
- `lang_pref`：JSON 字符串，语言 -> 占比
- `tag_pref`：JSON 字符串，标签 -> 占比（每题最多计入 2 个标签，减少噪声/维度）
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def parse_json_list(x: object) -> list[str]:
    """
    将「可能是列表/JSON 字符串/分隔字符串」的单元格解析为字符串列表。

    兼容的输入形态（常见于 CSV 的脏数据）：
    - Python list：`["dp", "math"]`
    - JSON 字符串：`'["dp","math"]'`
    - 其他字符串：`"dp, math"` / `"dp;math"` / `"dp | math"` 等
    - 空值：None / NaN / "nan" -> []
    """
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    s = str(x).strip()
    if not s or s.lower() == "nan":
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive student personas from CleanData tables.")
    parser.add_argument("--problems", default=str(ROOT / "CleanData/problems.csv"))
    parser.add_argument("--submissions", default=str(ROOT / "CleanData/submissions.csv"))
    parser.add_argument("--tags", default=str(ROOT / "CleanData/tags.csv"))
    parser.add_argument("--languages", default=str(ROOT / "CleanData/languages.csv"))
    parser.add_argument("--out", default=str(ROOT / "CleanData/students_derived.csv"))
    args = parser.parse_args()

    # 读取 CleanData 中间表：本脚本只做“派生画像”，不修改输入表。
    problems = pd.read_csv(args.problems)
    subs = pd.read_csv(args.submissions)
    tags = pd.read_csv(args.tags)
    langs = pd.read_csv(args.languages)

    # -------- 题目侧预处理：难度补全 + 标签列表化/规范化 --------
    # difficulty 可能为空/字符串；先转数值，随后用中位数补全（避免均值被极端值拉偏）。
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    # 仅保留 tags.csv 提供的词表中的标签，避免出现未登录词；并截断为最多 2 个标签以控维。
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    problems["tags_norm"] = problems["tags_list"].apply(
        lambda lst: [t for t in lst if t in tag_set][:2]
    )

    # -------- 提交侧预处理：AC 标记转为 0/1 --------
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

    # -------- 用户-题目聚合（User-Problem）--------
    # 对同一用户同一题的多次提交聚合：
    # - n_attempts：尝试次数（提交计数）
    # - solved：是否最终 AC（ac 的 max）
    up = (
        subs.groupby(["user_id", "problem_id"], as_index=False)
        .agg(n_attempts=("submission_id", "count"), solved=("ac", "max"))
        .merge(problems[["problem_id", "difficulty_filled", "tags_norm"]], on="problem_id", how="left")
    )

    # 合并后可能存在缺失题目元信息：难度用全局中位数兜底。
    up["difficulty_filled"] = up["difficulty_filled"].fillna(diff_median).astype(int)
    up["diff_norm"] = up["difficulty_filled"] / 10.0

    # -------- level：难度加权的完成度（0~1）--------
    # 直觉：高难题的 AC 对能力贡献更大；用“AC × 难度权重”的占比做归一化。
    num = (up["solved"] * up["diff_norm"]).groupby(up["user_id"]).sum()
    den = up["diff_norm"].groupby(up["user_id"]).sum()
    level = (num / (den + 1e-9)).reset_index(name="level")

    # -------- perseverance：平均每题尝试次数的归一化（0~1）--------
    # 使用对数缩放以缓和长尾；并用 95 分位做上界归一化，减少极端用户影响。
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

    # -------- lang_pref：用户语言偏好分布（JSON）--------
    # 仅统计 languages.csv 词表中出现的语言；按次数归一化为概率分布后序列化为 JSON。
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

    # -------- tag_pref：用户标签偏好分布（JSON）--------
    # 使用 up 表的 tags_norm（每题最多 2 个标签）累积计数并归一化为概率分布。
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

    # -------- 汇总输出：确保每个出现过提交的 user 都有一行画像 --------
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

    # 写出 CSV：utf-8-sig 便于在 Windows/Excel 下直接打开不乱码。
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Wrote", str(out_path), "rows=", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
