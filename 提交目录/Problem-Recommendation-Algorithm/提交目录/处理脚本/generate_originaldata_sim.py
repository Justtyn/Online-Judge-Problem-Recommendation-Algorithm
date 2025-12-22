"""
Utils/generate_originaldata_sim.py

用途
- 从 AcCodingData 的真实提交中提取模式，批量生成 OriginalData 的 submissions/students。
- 生成后对 OriginalData 做约束一致性检查，再复制产出 CleanData。
- 运行时提供阶段化进度可视与最终运行报告（RUN_REPORT）。

运行方法（常用）
1) 标准生成（10 万学生 + 300 万提交）：
   python Utils/generate_originaldata_sim.py --n-students 100000 --n-submissions 3000000
2) 调整语言占比上限/下限：
   python Utils/generate_originaldata_sim.py --lang-cap-c 0.60 --lang-min-java 0.03 \
     --lang-min-js 0.01 --lang-min-go 0.01 --n-students 100000 --n-submissions 3000000
3) 仅查看摘要（不写文件）：
   python Utils/generate_originaldata_sim.py --dry-run
4) 仅对比报告（不重新生成，只校验并对比）：
   python Utils/generate_originaldata_sim.py --report-only \
     --n-students 100000 --n-submissions 3000000
5) 调整长尾与 AC 前 WA：
   python Utils/generate_originaldata_sim.py --verdict-before-wa-mult 1.08 \
     --tail-boost-prob 0.25 --tail-boost-threshold 6 --tail-boost-factor 1.6 \
     --tail-boost-max 80 --n-students 100000 --n-submissions 3000000

输入/输出文件说明
- 输入：
  - AcCodingData/submissions.csv：真实提交，用于统计模式
  - AcCodingData/problems.csv：真实题库难度分布
- 输出：
  - OriginalData/students.csv
  - OriginalData/submissions.csv
  - 同时将 OriginalData 下 CSV 全量复制到 CleanData 目录

建模要点（来自真实数据分布的拟合/抽样）
- 用户活跃度长尾：采用对数正态分布，避免极端头部导致“单用户覆盖太多题”。
- 用户语言偏好强烈：多数用户单语言为主，全局语言占比匹配真实数据，并可纠偏。
- 题目难度选择：按真实提交的难度分布抽样。
- 通过行为：基于真实数据统计的 solved ratio / 首 AC 尝试次数 / AC 后额外尝试 / 未解尝试次数。
- Verdict 分布：分 AC 前 / AC 后 / 未解三段分别采样。
- exec_time_ms / mem_kb：零膨胀对数正态拟合 + 上界裁剪。

CLI 阶段可视化
- Step 1/4：统计真实提交分布（读取 AcCodingData/submissions.csv）
- Step 2/4：生成 OriginalData/students.csv 与 submissions.csv（显示进度/ETA）
- Step 3/4：一致性校验（显示校验进度）
- Step 4/4：复制到 CleanData

RUN_REPORT 报告解读（关键字段）
- students / submissions：生成的学生数与提交数
- unique_users_in_submissions：提交中出现的唯一用户数（应等于 students）
- unique_problems_in_submissions：提交中出现的题目数（应覆盖全部题目）
- ac_rate：总体 AC 比例
- language_dist：语言占比分布（用于检查 C 是否被限制、JAVA/JS/GO 是否补足）
- verdict_dist：判题分布（WA/TLE/RE/CE/AC）
- attempt_p50/p90/p99：attempt_no 分位数（提交维度）
- exec_time_p50/p95/p99 / mem_kb_p50/p95/p99：性能指标分位数（已考虑 0 值比例）
- exec_time_p_zero / mem_kb_p_zero：exec/mem 取 0 的比例（一般来自 CE 等）
- model_summary：生成时的关键参数（语言纠偏/活跃度参数等）
"""

import argparse
import bisect
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Optional, Union
from collections import Counter, defaultdict

CANON_LANGUAGES = ["Python", "C", "C++", "JS", "JAVA", "GO"]
CANON_VERDICTS = ["AC", "WA", "TLE", "RE", "CE"]

LANGUAGE_MAP = {
    "c++": "C++",
    "cpp": "C++",
    "g++": "C++",
    "c": "C",
    "python": "Python",
    "python2": "Python",
    "python3": "Python",
    "java": "JAVA",
    "js": "JS",
    "javascript": "JS",
    "node": "JS",
    "go": "GO",
    "golang": "GO",
}

VERDICT_MAP = {
    "AC": "AC",
    "WA": "WA",
    "TLE": "TLE",
    "CE": "CE",
    # Map other verdicts to the closest canonical bucket.
    "MLE": "RE",
    "REG": "RE",
    "REP": "RE",
    "IFNR": "RE",
    "OFNR": "RE",
    "PE": "WA",
    "OE": "WA",
}


def ensure_parent_dir(path: str) -> None:
    """确保输出文件所在目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def normalize_difficulty(value: Optional[Union[str, int]], *, default: int = 1) -> int:
    """将难度字段规整为 [1,10] 的整数。"""
    try:
        d = int(value)
    except (TypeError, ValueError):
        return default
    if d <= 0:
        d = 1
    if d > 10:
        d = 10
    return d


def map_language(raw: str) -> str:
    """将真实语言标签映射到 OriginalData 的标准语言集合。"""
    key = (raw or "").strip().lower()
    return LANGUAGE_MAP.get(key, "Python")


def map_verdict(raw: str) -> str:
    """将真实判题结果映射到 OriginalData 的标准 verdict 集合。"""
    key = (raw or "").strip().upper()
    return VERDICT_MAP.get(key, "RE")


def percentile(values: list[int], p: float) -> int:
    """返回 list 的 p 分位（基于下标截断）。"""
    if not values:
        return 0
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return values[idx]


def percentile_from_counter(counter: Counter[int], p: float) -> int:
    """从计数分布估算分位数。"""
    total = sum(counter.values())
    if total == 0:
        return 0
    threshold = total * p
    cum = 0
    for k in sorted(counter):
        cum += counter[k]
        if cum >= threshold:
            return k
    return max(counter)


class WeightedSampler:
    """基于累积分布的加权采样器。"""
    def __init__(self, items: Union[list[int], list[str]], weights: list[float]) -> None:
        if not items or len(items) != len(weights):
            raise ValueError("Invalid sampler inputs")
        self.items = list(items)
        self.cum_weights: list[float] = []
        total = 0.0
        for w in weights:
            w = float(w)
            if w < 0:
                w = 0.0
            total += w
            self.cum_weights.append(total)
        if total <= 0:
            raise ValueError("Total weight must be > 0")
        self.total = total

    def sample(self, rng: random.Random) -> Union[int, str]:
        x = rng.random() * self.total
        idx = bisect.bisect_left(self.cum_weights, x)
        return self.items[idx]


class StreamingLogStats:
    """
    轻量级的流式统计器，用于近似 zero-inflated lognormal 分布参数。
    - log_mean/log_M2 用于估计 log 空间的均值/方差
    - sample 为水库抽样，估算分位数（用于裁剪异常长尾）
    """
    def __init__(self, *, sample_size: int = 50000) -> None:
        self.sample_size = sample_size
        self.sample: list[int] = []
        self.count_total = 0
        self.count_positive = 0
        self.count_zero = 0
        self.log_mean = 0.0
        self.log_M2 = 0.0
        self.max_value = 0

    def update(self, value: int, rng: random.Random) -> None:
        self.count_total += 1
        # 0 或负值视为零膨胀部分。
        if value <= 0:
            self.count_zero += 1
            if value > self.max_value:
                self.max_value = value
            return

        if value > self.max_value:
            self.max_value = value

        self.count_positive += 1
        logv = math.log(value)
        delta = logv - self.log_mean
        self.log_mean += delta / self.count_positive
        delta2 = logv - self.log_mean
        self.log_M2 += delta * delta2

        # 水库采样：在内存可控情况下保留样本估计分位数。
        if self.sample_size <= 0:
            return
        if len(self.sample) < self.sample_size:
            self.sample.append(value)
        else:
            j = rng.randint(1, self.count_positive)
            if j <= self.sample_size:
                self.sample[j - 1] = value

    def finalize(self) -> dict[str, Union[float, int]]:
        """输出用于采样的参数与统计信息。"""
        if self.count_total == 0:
            return {
                "p_zero": 1.0,
                "mu": 0.0,
                "sigma": 1.0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "max": 0,
            }
        p_zero = self.count_zero / self.count_total
        if self.count_positive <= 1:
            mu = 0.0
            sigma = 1.0
        else:
            variance = self.log_M2 / self.count_positive
            mu = self.log_mean
            sigma = math.sqrt(max(variance, 1e-6))
        def adj_percentile(p: float) -> int:
            if p <= p_zero or not self.sample:
                return 0
            adj = (p - p_zero) / max(1e-9, (1.0 - p_zero))
            adj = min(max(adj, 0.0), 1.0)
            return percentile(self.sample, adj)

        p50 = adj_percentile(0.50)
        p95 = adj_percentile(0.95)
        p99 = adj_percentile(0.99)
        return {
            "p_zero": p_zero,
            "mu": mu,
            "sigma": sigma,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "max": self.max_value,
        }


class MetricSampler:
    """根据 StreamingLogStats 输出参数进行采样。"""
    def __init__(self, params: dict[str, Union[float, int]]) -> None:
        self.p_zero = float(params.get("p_zero", 0.0))
        self.mu = float(params.get("mu", 0.0))
        self.sigma = float(params.get("sigma", 1.0))
        self.p99 = int(params.get("p99", 0))
        self.max_value = int(params.get("max", 0))
        cap = max(self.p99, 1)
        if self.max_value > 0:
            cap = min(self.max_value, max(cap, 1) * 4)
        self.cap = max(cap, 0)

    def sample(self, rng: random.Random) -> int:
        """返回一个非负整数值（零膨胀 + lognormal + 截断）。"""
        if rng.random() < self.p_zero:
            return 0
        val = int(round(math.exp(rng.normalvariate(self.mu, self.sigma))))
        if val < 0:
            val = 0
        if self.cap > 0 and val > self.cap:
            val = self.cap
        return val


def load_problem_difficulties(path: str) -> dict[str, int]:
    """读取题库难度，用于为提交/对题目选择分配权重。"""
    diffs: dict[str, int] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("problem_id")
            if not pid:
                continue
            diffs[pid] = normalize_difficulty(row.get("difficulty"))
    return diffs


def estimate_zipf_alpha(counts: list[int]) -> float:
    """估计 Zipf 指数（仅用于参考，实际生成采用 lognormal 活跃度）。"""
    counts = sorted(counts, reverse=True)
    if not counts:
        return 1.0
    n = len(counts)
    max_rank = max(2, int(n * 0.8))
    xs = []
    ys = []
    for i in range(1, max_rank + 1):
        if counts[i - 1] <= 0:
            continue
        xs.append(math.log(i))
        ys.append(math.log(counts[i - 1]))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    alpha = -num / den if den else 1.0
    return max(0.5, min(alpha, 1.5))


def build_sampler_from_counter(counter: Counter) -> WeightedSampler:
    """将 Counter 转成可采样的 WeightedSampler。"""
    items = sorted(counter.keys())
    weights = [counter[k] for k in items]
    return WeightedSampler(items, weights)


def adjust_language_dist(
    dist: dict[str, float],
    *,
    cap_c: Optional[float],
    min_java: float,
    min_js: float,
    min_go: float,
) -> dict[str, float]:
    """
    对语言分布进行“纠偏”：
    - 限制 C 语言上限（避免占比过高）
    - 给 JAVA/JS/GO 设置最低占比（JS/GO 仍保持垫底）
    - 保证最终分布归一化到 1
    """
    base = {lang: max(0.0, float(dist.get(lang, 0.0))) for lang in CANON_LANGUAGES}

    if cap_c is not None and base["C"] > cap_c:
        base["C"] = cap_c

    min_map = {"JAVA": min_java, "JS": min_js, "GO": min_go}
    for lang, minv in min_map.items():
        base[lang] = max(base[lang], minv)

    total = sum(base.values())
    if total > 1.0 + 1e-9:
        # 先从高于最小值的语言中按比例回收，避免破坏最低占比约束。
        slack = {}
        for lang, val in base.items():
            min_floor = min_map.get(lang, 0.0)
            slack_amt = max(0.0, val - min_floor)
            if slack_amt > 0:
                slack[lang] = slack_amt
        excess = total - 1.0
        if slack:
            slack_total = sum(slack.values())
            for lang, slack_amt in slack.items():
                reduce = excess * (slack_amt / slack_total)
                base[lang] = max(min_map.get(lang, 0.0), base[lang] - reduce)
        else:
            for lang in base:
                base[lang] /= total
    elif total < 1.0 - 1e-9:
        # 按真实分布权重补足剩余占比，同时避免 C 超过 cap。
        remaining = 1.0 - total
        weights: dict[str, float] = {}
        for lang in CANON_LANGUAGES:
            weight = dist.get(lang, 0.0)
            if weight <= 0:
                weight = 0.01
            if lang == "C" and cap_c is not None and base["C"] >= cap_c - 1e-9:
                weight = 0.0
            weights[lang] = weight
        total_w = sum(weights.values())
        if total_w <= 0:
            weights = {"Python": 1.0, "C++": 1.0, "JAVA": 1.0}
            total_w = sum(weights.values())
        for lang, w in weights.items():
            if w <= 0:
                continue
            add = remaining * (w / total_w)
            if lang == "C" and cap_c is not None:
                cap_left = cap_c - base["C"]
                if cap_left <= 0:
                    continue
                add = min(add, cap_left)
            base[lang] += add

    total = sum(base.values())
    if total <= 0:
        return {"Python": 1.0}
    return {lang: base[lang] / total for lang in base}


def adjust_verdict_before_dist(dist: dict[str, float], *, wa_mult: float) -> dict[str, float]:
    """
    对 AC 前的 verdict 分布进行轻微纠偏。
    目的：在整体 AC 率偏高时，适度提高 WA 占比。
    """
    if not dist:
        return {"WA": 1.0}
    adjusted = dict(dist)
    if "WA" in adjusted:
        adjusted["WA"] *= max(0.01, wa_mult)
    total = sum(adjusted.values())
    if total <= 0:
        return {"WA": 1.0}
    return {k: v / total for k, v in adjusted.items()}


def apply_tail_boost(
    attempts: int,
    *,
    rng: random.Random,
    prob: float,
    threshold: int,
    factor: float,
    max_attempts: int,
) -> int:
    """
    对尝试次数的长尾进行增强（只在 attempts >= threshold 时生效）。
    """
    if attempts < threshold:
        return attempts
    if rng.random() >= prob:
        return attempts
    boosted = int(math.ceil(attempts * max(1.0, factor)))
    boosted = max(boosted, attempts + 1)
    return min(boosted, max_attempts)

def summarize_real_data(ac_dir: str, *, rng: random.Random) -> dict:
    """
    汇总真实提交数据的关键统计分布。

    注意：
    - 只扫描 AcCodingData/submissions.csv / problems.csv，不修改源数据。
    - 结果用于模拟生成时的抽样分布与参数估计。
    """
    problems_csv = os.path.join(ac_dir, "problems.csv")
    submissions_csv = os.path.join(ac_dir, "submissions.csv")

    if not os.path.exists(problems_csv):
        raise SystemExit(f"Missing {problems_csv}")
    if not os.path.exists(submissions_csv):
        raise SystemExit(f"Missing {submissions_csv}")

    prob_diff = load_problem_difficulties(problems_csv)

    lang_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    submission_diff_counts: Counter[int] = Counter()
    user_counts: Counter[str] = Counter()
    user_lang_counts: dict[str, Counter[str]] = defaultdict(Counter)

    pair_attempts: dict[tuple[str, str], int] = {}
    first_ac_attempt: dict[tuple[str, str], int] = {}
    attempt_no_counts: Counter[int] = Counter()

    exec_stats = {v: StreamingLogStats() for v in CANON_VERDICTS}
    mem_stats = {v: StreamingLogStats() for v in CANON_VERDICTS}
    exec_overall = StreamingLogStats()
    mem_overall = StreamingLogStats()

    start = time.time()
    total_rows = 0
    sys.stderr.write("Step 1/4 统计真实提交分布...\n")
    with open(submissions_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            raw_lang = row.get("language", "")
            raw_verdict = row.get("verdict", "")
            mapped_lang = map_language(raw_lang)
            mapped_verdict = map_verdict(raw_verdict)

            lang_counts[mapped_lang] += 1
            verdict_counts[mapped_verdict] += 1
            user_id = row.get("student_id", "")
            user_counts[user_id] += 1
            user_lang_counts[user_id][mapped_lang] += 1

            pid = row.get("problem_id", "")
            diff = prob_diff.get(pid)
            if diff is not None:
                submission_diff_counts[diff] += 1

            try:
                exec_ms = int(row.get("exec_time_ms", 0))
            except ValueError:
                exec_ms = 0
            try:
                mem_kb = int(row.get("memory_kb", 0))
            except ValueError:
                mem_kb = 0
            exec_stats[mapped_verdict].update(exec_ms, rng)
            mem_stats[mapped_verdict].update(mem_kb, rng)
            exec_overall.update(exec_ms, rng)
            mem_overall.update(mem_kb, rng)

            key = (user_id, pid)
            attempt = pair_attempts.get(key, 0) + 1
            pair_attempts[key] = attempt
            attempt_no_counts[attempt] += 1
            if raw_verdict == "AC" and key not in first_ac_attempt:
                first_ac_attempt[key] = attempt

            if total_rows % 1_000_000 == 0:
                elapsed = time.time() - start
                sys.stderr.write(
                    f"\rSummarizing real data: {total_rows} rows ({elapsed:.1f}s)"
                )
                sys.stderr.flush()

    sys.stderr.write("\n")

    pair_total_by_diff: Counter[int] = Counter()
    pair_solved_by_diff: Counter[int] = Counter()
    first_ac_by_diff: dict[int, Counter[int]] = defaultdict(Counter)
    extra_after_ac_by_diff: dict[int, Counter[int]] = defaultdict(Counter)
    unsolved_attempts_by_diff: dict[int, Counter[int]] = defaultdict(Counter)

    for (user_id, pid), total_attempts in pair_attempts.items():
        diff = prob_diff.get(pid)
        if diff is None:
            continue
        pair_total_by_diff[diff] += 1
        key = (user_id, pid)
        if key in first_ac_attempt:
            pair_solved_by_diff[diff] += 1
            first_ac = first_ac_attempt[key]
            first_ac_by_diff[diff][first_ac] += 1
            extra_after_ac_by_diff[diff][total_attempts - first_ac] += 1
        else:
            unsolved_attempts_by_diff[diff][total_attempts] += 1

    solved_ratio_by_diff = {
        diff: (pair_solved_by_diff[diff] / pair_total_by_diff[diff])
        for diff in pair_total_by_diff
    }

    # Verdict distribution by stage (before AC / after AC / unsolved).
    verdict_before: Counter[str] = Counter()
    verdict_after: Counter[str] = Counter()
    verdict_unsolved: Counter[str] = Counter()
    attempt_progress: dict[tuple[str, str], int] = {}

    with open(submissions_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row.get("student_id", ""), row.get("problem_id", ""))
            attempt = attempt_progress.get(key, 0) + 1
            attempt_progress[key] = attempt
            mapped = map_verdict(row.get("verdict", ""))
            first_ac = first_ac_attempt.get(key)
            if first_ac is None:
                verdict_unsolved[mapped] += 1
            else:
                if attempt < first_ac:
                    verdict_before[mapped] += 1
                elif attempt > first_ac:
                    verdict_after[mapped] += 1

    def normalize_counter(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    verdict_before_dist = normalize_counter(verdict_before)
    verdict_after_dist = normalize_counter(verdict_after)
    verdict_unsolved_dist = normalize_counter(verdict_unsolved)
    if "AC" in verdict_before_dist:
        verdict_before_dist.pop("AC", None)
    if "AC" in verdict_unsolved_dist:
        verdict_unsolved_dist.pop("AC", None)

    lang_dist = normalize_counter(lang_counts)
    verdict_dist = normalize_counter(verdict_counts)
    diff_dist = normalize_counter(submission_diff_counts)

    # 用户语言偏好：统计每位用户最大语言占比的平均值，作为单语言使用概率。
    top_shares = []
    for counts in user_lang_counts.values():
        total = sum(counts.values())
        if total == 0:
            continue
        top_shares.append(max(counts.values()) / total)
    primary_lang_prob = sum(top_shares) / len(top_shares) if top_shares else 0.95

    exec_params = {k: v.finalize() for k, v in exec_stats.items()}
    mem_params = {k: v.finalize() for k, v in mem_stats.items()}
    exec_overall_params = exec_overall.finalize()
    mem_overall_params = mem_overall.finalize()

    avg_attempts_per_pair = (
        sum(pair_attempts.values()) / len(pair_attempts) if pair_attempts else 2.5
    )
    zipf_alpha = estimate_zipf_alpha(list(user_counts.values()))
    # 对数正态的 sigma 控制长尾程度，避免极端头部造成“单人覆盖太多题”。
    log_counts = [math.log(c) for c in user_counts.values() if c > 0]
    if log_counts:
        mean_log = sum(log_counts) / len(log_counts)
        var_log = sum((x - mean_log) ** 2 for x in log_counts) / len(log_counts)
        log_sigma = math.sqrt(var_log)
    else:
        log_sigma = 1.0
    user_activity_sigma = min(max(log_sigma, 0.9), 1.3)

    ac_rate = verdict_counts.get("AC", 0) / total_rows if total_rows else 0.0
    attempt_p50 = percentile_from_counter(attempt_no_counts, 0.5)
    attempt_p90 = percentile_from_counter(attempt_no_counts, 0.9)
    attempt_p99 = percentile_from_counter(attempt_no_counts, 0.99)

    return {
        "language_dist": lang_dist,
        "verdict_dist": verdict_dist,
        "difficulty_dist": diff_dist,
        "solved_ratio_by_diff": solved_ratio_by_diff,
        "first_ac_by_diff": {k: dict(v) for k, v in first_ac_by_diff.items()},
        "extra_after_ac_by_diff": {k: dict(v) for k, v in extra_after_ac_by_diff.items()},
        "unsolved_attempts_by_diff": {k: dict(v) for k, v in unsolved_attempts_by_diff.items()},
        "verdict_before_dist": verdict_before_dist,
        "verdict_after_dist": verdict_after_dist,
        "verdict_unsolved_dist": verdict_unsolved_dist,
        "exec_params": exec_params,
        "mem_params": mem_params,
        "exec_overall": exec_overall_params,
        "mem_overall": mem_overall_params,
        "ac_rate": ac_rate,
        "attempt_p50": attempt_p50,
        "attempt_p90": attempt_p90,
        "attempt_p99": attempt_p99,
        "avg_attempts_per_pair": avg_attempts_per_pair,
        "zipf_alpha": zipf_alpha,
        "user_activity_sigma": user_activity_sigma,
        "primary_lang_prob": primary_lang_prob,
        "total_submissions": total_rows,
        "tail_boost_prob": 0.0,
        "tail_boost_threshold": 0,
        "tail_boost_factor": 1.0,
        "tail_boost_max": 0,
    }


def write_students_csv(path: str, *, n_students: int, dry_run: bool) -> None:
    """生成 students.csv 的空白字段（与项目预期字段一致）。"""
    ensure_parent_dir(path)
    if dry_run:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "level", "perseverance", "lang_pref", "tag_pref"])
        for user_id in range(1, n_students + 1):
            w.writerow([user_id, 0, 0, "{}", "{}"])


def load_problem_ids_by_diff(path: str) -> dict[int, list[int]]:
    """按难度组织题目 id，便于生成时匹配难度分布抽样。"""
    by_diff: dict[int, list[int]] = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("problem_id")
            if not pid:
                continue
            diff = normalize_difficulty(row.get("difficulty"))
            by_diff[diff].append(int(pid))
    return by_diff


def build_user_submission_counts(
    n_students: int, n_submissions: int, *, sigma: float, rng: random.Random
) -> list[int]:
    """
    生成每个用户的提交总量（对数正态分布 + 缩放）。

    目标：
    - 形成真实的“长尾”活跃度分布
    - 总提交数与用户数严格匹配
    """
    if n_submissions < n_students:
        raise SystemExit("n_submissions must be >= n_students to give each user >= 1 submission")
    mean_target = n_submissions / n_students
    mu = math.log(mean_target) - 0.5 * sigma * sigma
    raw = [max(1, int(rng.lognormvariate(mu, sigma))) for _ in range(n_students)]
    total = sum(raw)
    if total <= 0:
        raw = [1] * n_students
        total = n_students
    scale = n_submissions / total
    scaled = [max(1, int(x * scale)) for x in raw]
    current = sum(scaled)
    delta = n_submissions - current
    if delta > 0:
        for _ in range(delta):
            idx = rng.randrange(0, n_students)
            scaled[idx] += 1
    elif delta < 0:
        for _ in range(-delta):
            idx = rng.randrange(0, n_students)
            if scaled[idx] > 1:
                scaled[idx] -= 1
    rng.shuffle(scaled)
    return scaled


def generate_submissions_csv(
    submissions_csv: str,
    *,
    n_submissions: int,
    n_students: int,
    problem_ids_by_diff: dict[int, list[int]],
    summary: dict,
    rng: random.Random,
    dry_run: bool,
) -> None:
    """
    生成 submissions.csv。

    逻辑概要：
    1) 按难度分布挑选题目
    2) 按 solved_ratio 决定该题是否 AC
    3) 若 AC：先抽首个 AC 尝试次数，再抽 AC 后额外提交次数
    4) 若未解：抽未解尝试次数
    5) 按 AC 前/后/未解分布抽 verdict；语言按用户偏好抽样
    """
    ensure_parent_dir(submissions_csv)
    if dry_run:
        return

    diff_dist = summary["difficulty_dist"]
    solved_ratio_by_diff = summary["solved_ratio_by_diff"]
    primary_lang_prob = summary["primary_lang_prob"]

    lang_dist = summary["language_dist"]
    if not lang_dist:
        lang_dist = {"Python": 1.0}
    language_sampler = WeightedSampler(list(lang_dist.keys()), list(lang_dist.values()))

    def build_sampler_map(
        raw_map: dict[int, dict[int, int]], fallback: Counter[int]
    ) -> dict[int, WeightedSampler]:
        """按 diff 生成采样器；缺失 diff 时回退到全局采样器。"""
        samplers: dict[int, WeightedSampler] = {}
        for diff, counts in raw_map.items():
            if counts:
                samplers[diff] = build_sampler_from_counter(Counter(counts))
        fallback_sampler = build_sampler_from_counter(fallback)
        for diff in range(1, 11):
            if diff not in samplers:
                samplers[diff] = fallback_sampler
        return samplers

    first_ac_global = Counter()
    extra_after_global = Counter()
    unsolved_global = Counter()
    for counts in summary["first_ac_by_diff"].values():
        first_ac_global.update(counts)
    for counts in summary["extra_after_ac_by_diff"].values():
        extra_after_global.update(counts)
    for counts in summary["unsolved_attempts_by_diff"].values():
        unsolved_global.update(counts)

    if not first_ac_global:
        first_ac_global = Counter({1: 1})
    if not extra_after_global:
        extra_after_global = Counter({0: 1})
    if not unsolved_global:
        unsolved_global = Counter({1: 1})

    first_ac_sampler_map = build_sampler_map(summary["first_ac_by_diff"], first_ac_global)
    extra_after_sampler_map = build_sampler_map(
        summary["extra_after_ac_by_diff"], extra_after_global
    )
    unsolved_attempt_sampler_map = build_sampler_map(
        summary["unsolved_attempts_by_diff"], unsolved_global
    )

    if not diff_dist:
        diff_dist = {1: 1.0}
    diff_sampler = WeightedSampler(list(diff_dist.keys()), list(diff_dist.values()))

    verdict_before_dist = summary["verdict_before_dist"] or {"WA": 1.0}
    verdict_after_dist = summary["verdict_after_dist"] or {"AC": 1.0}
    verdict_unsolved_dist = summary["verdict_unsolved_dist"] or {"WA": 1.0}
    verdict_before_sampler = WeightedSampler(
        list(verdict_before_dist.keys()), list(verdict_before_dist.values())
    )
    verdict_after_sampler = WeightedSampler(
        list(verdict_after_dist.keys()), list(verdict_after_dist.values())
    )
    verdict_unsolved_sampler = WeightedSampler(
        list(verdict_unsolved_dist.keys()), list(verdict_unsolved_dist.values())
    )

    exec_samplers = {k: MetricSampler(v) for k, v in summary["exec_params"].items()}
    mem_samplers = {k: MetricSampler(v) for k, v in summary["mem_params"].items()}

    user_submission_counts = build_user_submission_counts(
        n_students, n_submissions, sigma=summary["user_activity_sigma"], rng=rng
    )

    all_problem_ids = []
    for ids in problem_ids_by_diff.values():
        all_problem_ids.extend(ids)
    if not all_problem_ids:
        raise SystemExit("No problems found in OriginalData/problems.csv")

    with open(submissions_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "submission_id",
                "user_id",
                "problem_id",
                "attempt_no",
                "language",
                "verdict",
                "ac",
                "exec_time_ms",
                "mem_kb",
            ]
        )

        submission_id = 1
        start = time.time()
        for user_id in range(1, n_students + 1):
            remaining = user_submission_counts[user_id - 1]
            used_problems: set[int] = set()
            primary_lang = language_sampler.sample(rng)

            while remaining > 0:
                diff = int(diff_sampler.sample(rng))
                pool = problem_ids_by_diff.get(diff) or all_problem_ids
                problem_id = 0

                for _ in range(50):
                    pid = pool[rng.randrange(0, len(pool))]
                    if pid not in used_problems:
                        problem_id = pid
                        break

                if not problem_id and len(used_problems) < len(pool):
                    unused = [pid for pid in pool if pid not in used_problems]
                    if unused:
                        problem_id = unused[rng.randrange(0, len(unused))]

                if not problem_id:
                    if len(used_problems) >= len(all_problem_ids):
                        used_problems.clear()
                    while True:
                        pid = all_problem_ids[rng.randrange(0, len(all_problem_ids))]
                        if pid not in used_problems:
                            problem_id = pid
                            break

                used_problems.add(problem_id)

                solved_prob = solved_ratio_by_diff.get(diff, 0.8)
                solved = rng.random() < solved_prob

                if solved:
                    first_ac = int(first_ac_sampler_map[diff].sample(rng))
                    extra_after = int(extra_after_sampler_map[diff].sample(rng))
                    extra_after = apply_tail_boost(
                        extra_after,
                        rng=rng,
                        prob=summary["tail_boost_prob"],
                        threshold=summary["tail_boost_threshold"],
                        factor=summary["tail_boost_factor"],
                        max_attempts=summary["tail_boost_max"],
                    )
                    total_attempts = max(1, first_ac + extra_after)
                else:
                    total_attempts = int(unsolved_attempt_sampler_map[diff].sample(rng))
                    total_attempts = apply_tail_boost(
                        total_attempts,
                        rng=rng,
                        prob=summary["tail_boost_prob"],
                        threshold=summary["tail_boost_threshold"],
                        factor=summary["tail_boost_factor"],
                        max_attempts=summary["tail_boost_max"],
                    )
                    total_attempts = max(1, total_attempts)
                    first_ac = 0

                if total_attempts > remaining:
                    total_attempts = remaining
                    if solved and first_ac > total_attempts:
                        solved = False
                        first_ac = 0

                for attempt_no in range(1, total_attempts + 1):
                    if solved and attempt_no == first_ac:
                        verdict = "AC"
                    elif solved and attempt_no > first_ac:
                        verdict = verdict_after_sampler.sample(rng)
                    elif solved:
                        verdict = verdict_before_sampler.sample(rng)
                    else:
                        verdict = verdict_unsolved_sampler.sample(rng)

                    if rng.random() < primary_lang_prob:
                        language = primary_lang
                    else:
                        language = language_sampler.sample(rng)

                    ac = 1 if verdict == "AC" else 0
                    exec_time = exec_samplers[verdict].sample(rng)
                    mem_kb = mem_samplers[verdict].sample(rng)

                    w.writerow(
                        [
                            submission_id,
                            user_id,
                            problem_id,
                            attempt_no,
                            language,
                            verdict,
                            ac,
                            exec_time,
                            mem_kb,
                        ]
                    )
                    submission_id += 1

                remaining -= total_attempts

            if user_id % 5000 == 0:
                elapsed = time.time() - start
                rate = submission_id / elapsed if elapsed > 0 else 0
                pct = min(100.0, (submission_id - 1) * 100.0 / n_submissions)
                eta = (n_submissions - (submission_id - 1)) / rate if rate > 0 else 0
                sys.stderr.write(
                    f"\rGenerated {submission_id - 1}/{n_submissions} "
                    f"({pct:.1f}%, {rate:.0f}/s, ETA {eta:.1f}s)"
                )
                sys.stderr.flush()

        sys.stderr.write("\n")


def validate_original_data(
    original_dir: str, *, expected_students: int, expected_submissions: int, rng: random.Random
) -> dict:
    """
    对 OriginalData 进行一致性与约束检查，并返回运行报告。

    约束检查包括：
    - user_id / problem_id 外键
    - submission_id 顺序性
    - attempt_no 连续性
    - verdict 与 ac 字段一致
    - exec_time_ms / mem_kb 非负
    """
    students_csv = os.path.join(original_dir, "students.csv")
    submissions_csv = os.path.join(original_dir, "submissions.csv")
    problems_csv = os.path.join(original_dir, "problems.csv")
    languages_csv = os.path.join(original_dir, "languages.csv")
    verdicts_csv = os.path.join(original_dir, "verdicts.csv")

    with open(students_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        user_ids_list = [row["user_id"] for row in r]
    if len(user_ids_list) != expected_students:
        raise SystemExit(
            f"students.csv count mismatch: {len(user_ids_list)} != {expected_students}"
        )
    user_ids = set(user_ids_list)
    if len(user_ids) != len(user_ids_list):
        raise SystemExit("students.csv has duplicate user_id")

    with open(problems_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        problem_ids = {row["problem_id"] for row in r if row.get("problem_id")}

    with open(languages_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        language_names = {row["name"] for row in r if row.get("name")}

    with open(verdicts_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        verdict_names = {row["name"] for row in r if row.get("name")}

    attempt_tracker: dict[tuple[str, str], int] = {}
    expected_submission_id = 1
    row_count = 0
    used_users: set[str] = set()
    used_problems: set[str] = set()
    verdict_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    attempt_counts: Counter[int] = Counter()
    ac_count = 0
    exec_stats = StreamingLogStats(sample_size=20000)
    mem_stats = StreamingLogStats(sample_size=20000)

    with open(submissions_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row_count += 1
            if row.get("submission_id") != str(expected_submission_id):
                raise SystemExit(f"submission_id not sequential at {expected_submission_id}")
            expected_submission_id += 1

            user_id = row.get("user_id", "")
            problem_id = row.get("problem_id", "")
            if user_id not in user_ids:
                raise SystemExit(f"submission references unknown user_id={user_id}")
            if problem_id not in problem_ids:
                raise SystemExit(f"submission references unknown problem_id={problem_id}")

            try:
                attempt_no = int(row.get("attempt_no", 0))
            except ValueError:
                raise SystemExit(f"invalid attempt_no for user_id={user_id} problem_id={problem_id}")
            if attempt_no <= 0:
                raise SystemExit(f"attempt_no must be >=1 for user_id={user_id} problem_id={problem_id}")

            key = (user_id, problem_id)
            last_attempt = attempt_tracker.get(key, 0)
            if attempt_no != last_attempt + 1:
                raise SystemExit(
                    f"attempt_no not sequential for user_id={user_id} problem_id={problem_id}"
                )
            attempt_tracker[key] = attempt_no

            language = row.get("language", "")
            verdict = row.get("verdict", "")
            ac = row.get("ac", "")
            if language not in language_names:
                raise SystemExit(f"unknown language={language}")
            if verdict not in verdict_names:
                raise SystemExit(f"unknown verdict={verdict}")
            if (verdict == "AC" and ac != "1") or (verdict != "AC" and ac != "0"):
                raise SystemExit(f"ac flag mismatch at submission_id={row.get('submission_id')}")

            try:
                exec_time = int(row.get("exec_time_ms", 0))
                mem_kb = int(row.get("mem_kb", 0))
            except ValueError:
                raise SystemExit(f"invalid exec_time_ms/mem_kb at submission_id={row.get('submission_id')}")
            if exec_time < 0 or mem_kb < 0:
                raise SystemExit(f"negative exec_time_ms/mem_kb at submission_id={row.get('submission_id')}")

            used_users.add(user_id)
            used_problems.add(problem_id)
            verdict_counts[verdict] += 1
            language_counts[language] += 1
            attempt_counts[attempt_no] += 1
            if verdict == "AC":
                ac_count += 1
            exec_stats.update(exec_time, rng)
            mem_stats.update(mem_kb, rng)

            if row_count % 1_000_000 == 0:
                pct = min(100.0, row_count * 100.0 / expected_submissions)
                sys.stderr.write(f"\rValidating: {row_count} rows ({pct:.1f}%)")
                sys.stderr.flush()

    if row_count != expected_submissions:
        raise SystemExit(
            f"submissions.csv count mismatch: {row_count} != {expected_submissions}"
        )

    sys.stderr.write("\n")

    def normalize(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    exec_report = exec_stats.finalize()
    mem_report = mem_stats.finalize()

    report = {
        "students": expected_students,
        "submissions": row_count,
        "unique_users_in_submissions": len(used_users),
        "unique_problems_in_submissions": len(used_problems),
        "ac_rate": ac_count / row_count if row_count else 0.0,
        "language_dist": normalize(language_counts),
        "verdict_dist": normalize(verdict_counts),
        "attempt_p50": percentile_from_counter(attempt_counts, 0.5),
        "attempt_p90": percentile_from_counter(attempt_counts, 0.9),
        "attempt_p99": percentile_from_counter(attempt_counts, 0.99),
        "exec_time_p50": exec_report.get("p50", 0),
        "exec_time_p95": exec_report.get("p95", 0),
        "exec_time_p99": exec_report.get("p99", 0),
        "mem_kb_p50": mem_report.get("p50", 0),
        "mem_kb_p95": mem_report.get("p95", 0),
        "mem_kb_p99": mem_report.get("p99", 0),
        "exec_time_p_zero": exec_report.get("p_zero", 0.0),
        "mem_kb_p_zero": mem_report.get("p_zero", 0.0),
    }
    return report


def copy_to_clean_data(original_dir: str, clean_dir: str) -> None:
    """将 OriginalData 复制到 CleanData，保证后续流程使用统一入口。"""
    ensure_parent_dir(os.path.join(clean_dir, "placeholder.txt"))
    for name in ["students.csv", "submissions.csv", "problems.csv", "languages.csv", "verdicts.csv", "tags.csv"]:
        src = os.path.join(original_dir, name)
        dst = os.path.join(clean_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f"Missing {src} for CleanData export")
        shutil.copyfile(src, dst)


def diff_numeric(real_value: float, gen_value: float) -> float:
    return gen_value - real_value


def diff_map(real_map: dict[str, float], gen_map: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(real_map) | set(gen_map))
    return {k: gen_map.get(k, 0.0) - real_map.get(k, 0.0) for k in keys}


def build_real_report(summary: dict) -> dict:
    return {
        "submissions": summary.get("total_submissions", 0),
        "ac_rate": summary.get("ac_rate", 0.0),
        "language_dist": summary.get("language_dist_raw", summary.get("language_dist", {})),
        "verdict_dist": summary.get("verdict_dist", {}),
        "attempt_p50": summary.get("attempt_p50", 0),
        "attempt_p90": summary.get("attempt_p90", 0),
        "attempt_p99": summary.get("attempt_p99", 0),
        "exec_time_p50": summary.get("exec_overall", {}).get("p50", 0),
        "exec_time_p95": summary.get("exec_overall", {}).get("p95", 0),
        "exec_time_p99": summary.get("exec_overall", {}).get("p99", 0),
        "mem_kb_p50": summary.get("mem_overall", {}).get("p50", 0),
        "mem_kb_p95": summary.get("mem_overall", {}).get("p95", 0),
        "mem_kb_p99": summary.get("mem_overall", {}).get("p99", 0),
        "exec_time_p_zero": summary.get("exec_overall", {}).get("p_zero", 0.0),
        "mem_kb_p_zero": summary.get("mem_overall", {}).get("p_zero", 0.0),
    }


def build_comparison_report(real_report: dict, gen_report: dict) -> dict:
    return {
        "ac_rate_diff": diff_numeric(real_report.get("ac_rate", 0.0), gen_report.get("ac_rate", 0.0)),
        "language_dist_diff": diff_map(
            real_report.get("language_dist", {}), gen_report.get("language_dist", {})
        ),
        "verdict_dist_diff": diff_map(
            real_report.get("verdict_dist", {}), gen_report.get("verdict_dist", {})
        ),
        "attempt_p50_diff": diff_numeric(
            real_report.get("attempt_p50", 0), gen_report.get("attempt_p50", 0)
        ),
        "attempt_p90_diff": diff_numeric(
            real_report.get("attempt_p90", 0), gen_report.get("attempt_p90", 0)
        ),
        "attempt_p99_diff": diff_numeric(
            real_report.get("attempt_p99", 0), gen_report.get("attempt_p99", 0)
        ),
        "exec_time_p50_diff": diff_numeric(
            real_report.get("exec_time_p50", 0), gen_report.get("exec_time_p50", 0)
        ),
        "exec_time_p95_diff": diff_numeric(
            real_report.get("exec_time_p95", 0), gen_report.get("exec_time_p95", 0)
        ),
        "exec_time_p99_diff": diff_numeric(
            real_report.get("exec_time_p99", 0), gen_report.get("exec_time_p99", 0)
        ),
        "mem_kb_p50_diff": diff_numeric(
            real_report.get("mem_kb_p50", 0), gen_report.get("mem_kb_p50", 0)
        ),
        "mem_kb_p95_diff": diff_numeric(
            real_report.get("mem_kb_p95", 0), gen_report.get("mem_kb_p95", 0)
        ),
        "mem_kb_p99_diff": diff_numeric(
            real_report.get("mem_kb_p99", 0), gen_report.get("mem_kb_p99", 0)
        ),
        "exec_time_p_zero_diff": diff_numeric(
            real_report.get("exec_time_p_zero", 0.0),
            gen_report.get("exec_time_p_zero", 0.0),
        ),
        "mem_kb_p_zero_diff": diff_numeric(
            real_report.get("mem_kb_p_zero", 0.0), gen_report.get("mem_kb_p_zero", 0.0)
        ),
    }


def main() -> int:
    """CLI 入口，串联：汇总真实数据 -> 生成 -> 校验 -> 复制 -> 报告输出。"""
    parser = argparse.ArgumentParser(
        description="Generate OriginalData submissions/students from AcCodingData patterns, validate, and export to CleanData."
    )
    parser.add_argument("--ac-dir", default="AcCodingData", help="Path to real AcCodingData directory")
    parser.add_argument(
        "--original-dir", default="OriginalData", help="Output directory for OriginalData"
    )
    parser.add_argument("--clean-dir", default="CleanData", help="Output directory for CleanData")
    parser.add_argument("--n-students", type=int, default=100000, help="Number of students")
    parser.add_argument(
        "--n-submissions", type=int, default=3000000, help="Number of submissions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lang-cap-c",
        type=float,
        default=0.60,
        help="Max share of language C (0-1).",
    )
    parser.add_argument(
        "--lang-min-java",
        type=float,
        default=0.03,
        help="Min share of language JAVA (0-1).",
    )
    parser.add_argument(
        "--lang-min-js",
        type=float,
        default=0.01,
        help="Min share of language JS (0-1).",
    )
    parser.add_argument(
        "--lang-min-go",
        type=float,
        default=0.01,
        help="Min share of language GO (0-1).",
    )
    parser.add_argument(
        "--verdict-before-wa-mult",
        type=float,
        default=1.08,
        help="Multiply WA weight for verdicts before AC (>=1 means more WA).",
    )
    parser.add_argument(
        "--primary-lang-prob",
        type=float,
        default=None,
        help="Override primary language probability (0-1).",
    )
    parser.add_argument(
        "--tail-boost-prob",
        type=float,
        default=0.25,
        help="Probability to boost tail attempts (0-1).",
    )
    parser.add_argument(
        "--tail-boost-threshold",
        type=int,
        default=6,
        help="Only boost attempts >= this threshold.",
    )
    parser.add_argument(
        "--tail-boost-factor",
        type=float,
        default=1.6,
        help="Tail boost multiplicative factor.",
    )
    parser.add_argument(
        "--tail-boost-max",
        type=int,
        default=80,
        help="Upper bound for boosted attempts.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only analyze/validate existing OriginalData and output comparison report.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write any files")
    args = parser.parse_args()

    if args.n_students <= 0:
        raise SystemExit("--n-students must be > 0")
    if args.n_submissions <= 0:
        raise SystemExit("--n-submissions must be > 0")

    rng = random.Random(args.seed)

    summary = summarize_real_data(args.ac_dir, rng=rng)
    summary["language_dist_raw"] = summary.get("language_dist", {})
    summary["language_dist_adjusted"] = adjust_language_dist(
        summary["language_dist_raw"],
        cap_c=args.lang_cap_c,
        min_java=args.lang_min_java,
        min_js=args.lang_min_js,
        min_go=args.lang_min_go,
    )
    summary["language_dist"] = summary["language_dist_adjusted"]
    summary["verdict_before_dist"] = adjust_verdict_before_dist(
        summary.get("verdict_before_dist", {}), wa_mult=args.verdict_before_wa_mult
    )
    if args.primary_lang_prob is not None:
        summary["primary_lang_prob"] = min(max(args.primary_lang_prob, 0.0), 1.0)
    summary["tail_boost_prob"] = min(max(args.tail_boost_prob, 0.0), 1.0)
    summary["tail_boost_threshold"] = max(1, args.tail_boost_threshold)
    summary["tail_boost_factor"] = max(1.0, args.tail_boost_factor)
    summary["tail_boost_max"] = max(1, args.tail_boost_max)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "students": args.n_students,
                    "submissions": args.n_submissions,
                    "zipf_alpha": summary["zipf_alpha"],
                    "primary_lang_prob": summary["primary_lang_prob"],
                    "language_dist": summary["language_dist"],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.report_only:
        sys.stderr.write("Report-only 模式：跳过生成，直接校验并对比...\n")
        report = validate_original_data(
            args.original_dir,
            expected_students=args.n_students,
            expected_submissions=args.n_submissions,
            rng=rng,
        )
        real_report = build_real_report(summary)
        report["real_data"] = real_report
        report["comparison"] = build_comparison_report(real_report, report)
        report["model_summary"] = {
            "primary_lang_prob": summary["primary_lang_prob"],
            "user_activity_sigma": summary["user_activity_sigma"],
            "zipf_alpha_est": summary["zipf_alpha"],
            "language_cap_c": args.lang_cap_c,
            "language_min_java": args.lang_min_java,
            "language_min_js": args.lang_min_js,
            "language_min_go": args.lang_min_go,
            "language_dist_raw": summary["language_dist_raw"],
            "language_dist_adjusted": summary["language_dist_adjusted"],
        }
        print("OK")
        print("RUN_REPORT")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    original_students_csv = os.path.join(args.original_dir, "students.csv")
    original_submissions_csv = os.path.join(args.original_dir, "submissions.csv")
    original_problems_csv = os.path.join(args.original_dir, "problems.csv")

    if not os.path.exists(original_problems_csv):
        raise SystemExit(f"Missing {original_problems_csv}")

    sys.stderr.write("Step 2/4 生成 OriginalData/students.csv 与 submissions.csv...\n")
    write_students_csv(original_students_csv, n_students=args.n_students, dry_run=False)
    problem_ids_by_diff = load_problem_ids_by_diff(original_problems_csv)
    generate_submissions_csv(
        original_submissions_csv,
        n_submissions=args.n_submissions,
        n_students=args.n_students,
        problem_ids_by_diff=problem_ids_by_diff,
        summary=summary,
        rng=rng,
        dry_run=False,
    )

    sys.stderr.write("Step 3/4 校验 OriginalData 一致性...\n")
    report = validate_original_data(
        args.original_dir,
        expected_students=args.n_students,
        expected_submissions=args.n_submissions,
        rng=rng,
    )
    real_report = build_real_report(summary)
    report["real_data"] = real_report
    report["comparison"] = build_comparison_report(real_report, report)
    report["model_summary"] = {
        "primary_lang_prob": summary["primary_lang_prob"],
        "user_activity_sigma": summary["user_activity_sigma"],
        "zipf_alpha_est": summary["zipf_alpha"],
        "language_cap_c": args.lang_cap_c,
        "language_min_java": args.lang_min_java,
        "language_min_js": args.lang_min_js,
        "language_min_go": args.lang_min_go,
        "verdict_before_wa_mult": args.verdict_before_wa_mult,
        "tail_boost_prob": summary["tail_boost_prob"],
        "tail_boost_threshold": summary["tail_boost_threshold"],
        "tail_boost_factor": summary["tail_boost_factor"],
        "tail_boost_max": summary["tail_boost_max"],
        "language_dist_raw": summary["language_dist_raw"],
        "language_dist_adjusted": summary["language_dist_adjusted"],
    }
    sys.stderr.write("Step 4/4 复制到 CleanData...\n")
    copy_to_clean_data(args.original_dir, args.clean_dir)

    print("OK")
    print("RUN_REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
