"""
02_build_features.py

从 `CleanData/` 构造训练样本（`FeatureData/train_samples.csv`）。

核心思路：
- 以每一条 submission 作为一条样本（时序样本）。
- 特征全部使用“当前 submission 之前”的历史信息（避免用未来信息泄漏）。
- 结合题目静态信息（难度、标签）与用户动态画像（水平/坚持度/偏好语言/偏好标签）。

输出字段大致包含：
- 基础键：submission_id / user_id / problem_id / attempt_no
- 题目：difficulty_filled（缺失用中位数填充），以及派生的 diff_norm（内部使用）
- 用户动态特征（随时间变化）：
  - level：历史已 AC 题目的难度强度 / 历史尝试过题目的难度强度
  - perseverance：历史总提交数 / 历史尝试题数（做 log1p 后按全局 P95 归一到 [0,1]）
  - lang_match：当前语言在历史中出现的比例
  - tag_match：当前题目前两标签在历史中的出现比例（取平均）
- 稀疏特征：语言 one-hot、标签 multi-hot（每题最多取 2 个标签）
- 标签：ac（0/1）
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# 输入：由 01_derive_students.py（或其它清洗脚本）生成的 CleanData 目录
PROBLEMS = ROOT / "CleanData/problems.csv"
SUBS = ROOT / "CleanData/submissions.csv"
SUBS_COMPAT = ROOT / "CleanData/submissions_clean.csv"
TAGS = ROOT / "CleanData/tags.csv"
LANGS = ROOT / "CleanData/languages.csv"

# 输出：模型训练使用的样本表
OUT = ROOT / "FeatureData/train_samples.csv"

# 固定随机种子，保证可复现（当前脚本基本不依赖随机，但保持一致性）
RANDOM_SEED = 42


def first_existing_path(*candidates: Path) -> Path:
    """返回第一个存在的路径（用于兼容历史文件名）。"""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到输入文件，候选路径：{[str(c) for c in candidates]!r}")


def parse_json_list(x: object) -> list[str]:
    """
    将 CSV 单元格解析为字符串列表。

    支持几类常见脏数据格式：
    - 真正的 Python list（例如上游已解析）
    - JSON 字符串（例如 '["dp", "greedy"]'）
    - 逗号/分号/竖线分隔字符串（例如 'dp, greedy' / 'dp;greedy' / 'dp | greedy'）
    - None / NaN / 空字符串
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
    np.random.seed(RANDOM_SEED)

    # 读取清洗后的输入数据
    problems = pd.read_csv(PROBLEMS)
    subs = pd.read_csv(first_existing_path(SUBS, SUBS_COMPAT), low_memory=False)
    tags = pd.read_csv(TAGS)
    langs = pd.read_csv(LANGS)

    # 构建词表：用于后续 one-hot / multi-hot 的列顺序固定（保证训练可复现）
    tag_vocab = tags["tag_name"].astype(str).tolist()
    tag_set = set(tag_vocab)
    lang_vocab = sorted(set(langs["name"].astype(str).tolist()))
    lang_set = set(lang_vocab)

    # 题目表预处理：难度缺失用中位数填充，并归一化到 [0,1]（按 10 分制假设）
    problems["problem_id"] = pd.to_numeric(problems["problem_id"], errors="coerce").astype(int)
    problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
    diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
    problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
    problems["diff_norm"] = problems["difficulty_filled"].astype(float) / 10.0
    problems["tags_list"] = problems["tags"].apply(parse_json_list)

    # 每题只保留最多 2 个标签：降低维度/噪声，并让 tag_match 更稳定
    problems["tags2"] = problems["tags_list"].apply(lambda lst: [t for t in lst if t in tag_set][:2])

    # 映射表：submission 里只有 problem_id，需要快速查题目静态属性
    pid_to_diff = dict(zip(problems["problem_id"].astype(int), problems["difficulty_filled"].astype(int)))
    pid_to_diffnorm = dict(zip(problems["problem_id"].astype(int), problems["diff_norm"].astype(float)))
    pid_to_tags2 = dict(zip(problems["problem_id"].astype(int), problems["tags2"].tolist()))

    # 提交表字段清洗：保证类型稳定（便于后续 groupby/数组计算）
    subs["submission_id"] = pd.to_numeric(subs["submission_id"], errors="coerce").astype(int)
    subs["user_id"] = pd.to_numeric(subs["user_id"], errors="coerce").astype(int)
    subs["problem_id"] = pd.to_numeric(subs["problem_id"], errors="coerce").astype(int)
    subs["attempt_no"] = pd.to_numeric(subs["attempt_no"], errors="coerce").fillna(1).astype(int)
    subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)
    subs["language"] = subs.get("language", "").astype(str).fillna("")

    # 时序排序：对每个用户按 submission_id 从早到晚处理，确保“只看过去”
    subs = subs.sort_values(["user_id", "submission_id"]).reset_index(drop=True)

    # 用全局统计对 perseverance 做归一化：避免不同数据集规模导致尺度飘移
    # perseverance 的原始度量是 avg_attempts = total_subs / unique_problems
    user_total = subs.groupby("user_id").size()
    user_unique_prob = subs.groupby("user_id")["problem_id"].nunique()
    avg_attempts = (user_total / user_unique_prob.replace(0, np.nan)).dropna().values.astype(float)
    p95 = float(np.percentile(avg_attempts, 95)) if len(avg_attempts) else 1.0
    denom_p = math.log1p(p95) if p95 > 0 else 1.0

    n = len(subs)

    # 预分配数组（比在 DataFrame 里逐行写入更快）
    level_arr = np.zeros((n,), dtype=np.float32)
    pers_arr = np.zeros((n,), dtype=np.float32)
    lang_match_arr = np.zeros((n,), dtype=np.float32)
    tag_match_arr = np.zeros((n,), dtype=np.float32)

    # 题目静态信息对齐到 submission 序列（按 subs 的行顺序）
    tags2_list = subs["problem_id"].map(lambda pid: pid_to_tags2.get(int(pid), [])).to_list()
    diff_filled = subs["problem_id"].map(lambda pid: pid_to_diff.get(int(pid), diff_median)).astype(int)
    diff_norm = subs["problem_id"].map(lambda pid: pid_to_diffnorm.get(int(pid), diff_median / 10.0)).astype(float)

    tag_to_j = {t: j for j, t in enumerate(tag_vocab)}

    # 逐用户扫描：在每个 submission i 处，特征只由 i 之前的历史累积得到
    for uid, idx in subs.groupby("user_id", sort=False).groups.items():
        idx_list = idx.tolist()

        # 用户历史状态（随着提交滚动更新）
        problems_attempted: set[int] = set()
        problems_solved: set[int] = set()
        denom_sum = 0.0
        num_sum = 0.0
        total_subs = 0
        lang_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        total_tag = 0

        for i in idx_list:
            pid = int(subs.at[i, "problem_id"])
            lang = str(subs.at[i, "language"] or "")
            ac = int(subs.at[i, "ac"])

            # level：历史已 AC 的难度强度 / 历史尝试过的难度强度（都不含当前 submission）
            if denom_sum > 0:
                level_arr[i] = float(num_sum / (denom_sum + 1e-9))
            else:
                level_arr[i] = 0.0

            # perseverance：历史总提交数 / 历史尝试题数，做 log1p 并按全局 P95 归一化
            if total_subs > 0 and len(problems_attempted) > 0 and denom_p > 0:
                avg = float(total_subs) / float(len(problems_attempted))
                pers_arr[i] = float(max(0.0, min(1.0, math.log1p(avg) / denom_p)))
            else:
                pers_arr[i] = 0.0

            # lang_match：当前语言在历史提交中出现的比例（越高表示越“常用该语言”）
            if total_subs > 0 and lang in lang_set:
                lang_match_arr[i] = float(lang_counts.get(lang, 0) / total_subs)
            else:
                lang_match_arr[i] = 0.0

            # tag_match：当前题目的（最多 2 个）标签在历史中出现的比例（取均值）
            t2 = tags2_list[i]
            if t2 and total_tag > 0:
                tag_match_arr[i] = float(sum(tag_counts.get(t, 0) / total_tag for t in t2) / len(t2))
            else:
                tag_match_arr[i] = 0.0

            # ---- 从这里开始：把“当前 submission”更新进历史状态（供后续 submission 使用）----
            total_subs += 1
            if lang in lang_set:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            # 题目第一次被尝试时，才把该题的难度/标签计入分母（避免同题反复提交重复计权）
            if pid not in problems_attempted:
                problems_attempted.add(pid)
                denom_sum += float(diff_norm.iat[i])
                for t in t2:
                    if t in tag_to_j:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                        total_tag += 1

            # 题目第一次 AC 时，才把该题的难度计入分子（同题重复 AC 不重复计权）
            if ac == 1 and pid not in problems_solved:
                problems_solved.add(pid)
                num_sum += float(diff_norm.iat[i])

    # language one-hot：保持列全集（对没出现的语言补 0），保证下游训练列一致
    lang_ohe = pd.get_dummies(subs["language"], prefix="lang")
    for l in lang_vocab:
        c = f"lang_{l}"
        if c not in lang_ohe.columns:
            lang_ohe[c] = 0
    lang_ohe = lang_ohe[[f"lang_{l}" for l in lang_vocab]]

    # tag multi-hot：每条 submission 对应题目的 tags2（最多 2 个标签）
    tag_mh = np.zeros((n, len(tag_vocab)), dtype=np.uint8)
    for i, t2 in enumerate(tags2_list):
        if not t2:
            continue
        for t in t2:
            j = tag_to_j.get(t)
            if j is not None:
                tag_mh[i, j] = 1
    tag_mh_df = pd.DataFrame(tag_mh, columns=[f"tag_{t}" for t in tag_vocab])

    # 汇总输出：features + 稀疏编码 + label
    out = pd.concat(
        [
            subs[["submission_id", "user_id", "problem_id", "attempt_no"]].astype(int),
            pd.DataFrame(
                {
                    "difficulty_filled": diff_filled.astype(int),
                    "level": level_arr.astype(float),
                    "perseverance": pers_arr.astype(float),
                    "lang_match": lang_match_arr.astype(float),
                    "tag_match": tag_match_arr.astype(float),
                }
            ),
            lang_ohe.astype(int),
            tag_mh_df.astype(int),
            subs[["ac"]].astype(int),
        ],
        axis=1,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("Wrote", str(OUT), "rows=", len(out), "cols=", out.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
