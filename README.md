# Online Judge 题目推荐算法（OJ Problem Recommendation）

本仓库实现了一个面向 Online Judge 的“题目推荐算法”端到端流水线：从行为日志反推学生画像、构造训练样本、训练/评估 AC 预测模型，到基于预测概率的 Top‑K 推荐与推荐评估，并提供本地 Web 页面用于展示图表与交互式自定义推荐。

- 任务定义：把“某次提交是否 AC”建模为二分类问题，输出 `P(AC)`；推荐阶段用 `P(AC)` 作为排序分数（并支持“成长型推荐”概率区间过滤）

---

## 目录结构（数据流）

> 建议流水线：校验数据 → 派生画像 → 构造样本 → 训练评估 → Top‑K 推荐评估 → 产出图表与报告素材

```
.
├── OriginalData/     # 原始数据/素材（默认只读，例如题库压缩包）
├── CleanData/        # 输入/派生的干净表（本仓库默认从这里读取）
├── FeatureData/      # 可直接训练的样本表（X + y）
├── Models/           # 模型评估指标（可扩展保存模型文件）
├── Reports/          # 报告产物（图表/表格/推荐结果/诊断）
│   ├── fig/          # 所有图表（fig_*.png）
│   ├── reco/         # 推荐输出（recommendations_topk*.csv, reco_metrics.csv）
│   ├── diag/         # 诊断输出（diag_*.txt/csv）
│   ├── compare/      # strict vs leaky 对比产物（compare_*.csv + md）
│   └── validate/     # 数据校验报告（validate_report.txt）
├── Utils/            # 数据校验、抓取/解析、标注、模拟数据等小工具
├── WebApp/           # 本地 Web：展示图表 + 自定义推荐
├── 01_derive_students.py
├── 02_build_features.py
├── 03_train_eval.py
├── 04_recommend_eval.py
└── 05_make_eda_plots.py
```

说明：
- `OriginalData/` 在规范中作为“原始输入目录”（默认只读），但可能不会随仓库一并提交（数据通常较大/含隐私）。本仓库当前流水线默认直接读 `CleanData/`。
- 如果你没有真实 submissions，可用 `Utils/generate_originaldata_sim.py` 基于现有 `CleanData/problems.csv` 生成最小可跑通的模拟数据（students/languages/verdicts/submissions，见下文）。

---

## 快速开始（本地复现实验）

### 1）准备环境

建议 Python 版本：`>= 3.10`（本仓库脚本以 `pandas / numpy / scikit-learn / matplotlib` 为主）。

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
# （可选）开发/测试/Notebook
# python -m pip install -r requirements-optional.txt
```

### 2）准备输入数据（`CleanData/*.csv`）

最小输入集合（必须存在）：
- `CleanData/problems.csv`
- `CleanData/submissions.csv`
- `CleanData/students.csv`（可为空画像占位，但必须能提供 `user_id` 列）
- `CleanData/tags.csv`
- `CleanData/languages.csv`
- `CleanData/verdicts.csv`

先跑一次一致性校验（强烈建议）：

```bash
python Utils/validate_originaldata.py
# 如你的 problems.tags 是逗号分隔而不是 JSON 数组，可加：
# python Utils/validate_originaldata.py --accept-csv-tags
```

输出报告（可选）：

```bash
python Utils/validate_originaldata.py --report Reports/validate/validate_report.txt
```

### 2.5）诊断/校验：为什么推荐偏“全是难度 1”

当你发现某些学生提交了不少中等题，但 Top‑K 却被低难度“刷屏”时，可用诊断脚本输出量化证据（含候选集按难度的 `P(AC)` 分布、Top‑K 列表、以及可失败的校验 flags）：

```bash
# 单个学生：输出文本报告 + 图表到 Reports/diag/ 与 Reports/fig/
python Utils/diagnose_reco_bias.py --user-id 104 --cutoff-pct 0.50 --min-p 0.40 --max-p 0.70 --k 10 --plot

# 单个学生：命中指定 flags 则退出码=2（便于接 CI/批处理）
python Utils/diagnose_reco_bias.py --user-id 104 --fail-on easy_bias_vs_history,score_plateau_topk

# 批量扫描（默认扫所有有 submissions 的用户）：输出 CSV；可用 --fail-if-any 作为“全局校验”
python Utils/diagnose_reco_bias.py --scan --max-users 300 --fail-if-any
```

默认产物：
- `Reports/diag/diag_user_<user_id>.txt`
- `Reports/diag/diag_scan_users.csv`（scan 模式）
- `Reports/fig/fig_diag_user_<user_id>_reco_diff_hist.png`
- `Reports/fig/fig_diag_user_<user_id>_candidate_p_by_diff.png`

### 3）运行推荐算法流水线（03 → 06）

说明：以下脚本默认以仓库根目录为基准解析 `CleanData/FeatureData/Models/Reports` 路径，因此可以在任意工作目录运行。

```bash
python 01_derive_students.py      # 画像：CleanData/students_derived.csv
python 02_build_features.py       # 样本：FeatureData/train_samples.csv
python 03_train_eval.py           # 模型：Models/metrics.csv + Reports/fig/fig_cm_*.png
python 04_recommend_eval.py       # 推荐：Reports/reco/recommendations_topk.csv + Reports/reco/reco_metrics.csv
python 05_make_eda_plots.py       # EDA：Reports/fig/fig_*.png（用于写报告的“分布合理性”图）
```

关键产物（路径固定，便于引用）：
- `CleanData/students_derived.csv`：学生画像（`level / perseverance / lang_pref / tag_pref`）
- `FeatureData/train_samples.csv`：训练样本（可直接喂给 sklearn）
- `Models/metrics.csv`：分类模型对比指标（Accuracy/Precision/Recall/F1）
- `Reports/reco/recommendations_topk.csv`：Top‑K 推荐结果
- `Reports/reco/reco_metrics.csv`：Hit@K / Precision@K 等推荐评估指标
- `Reports/fig/fig_*.png`：所有图表（EDA + 混淆矩阵 + Hit@K 曲线等）

---

## 本地 Web（图表展示 + 自定义推荐）

Web 页面会：
1) 自动展示 `Reports/fig/` 下所有 `fig_*.png`（并按“数据层/训练层/推荐层”分区）  
2) 提供一个“自定义学生画像”的表单（`level/perseverance/语言偏好/标签偏好`），在线生成 Top‑K 推荐列表

启动：

```bash
python WebApp/server.py --port 8000
```

打开：
- `http://127.0.0.1:8000`（图表总览）
- `http://127.0.0.1:8000/custom`（自定义推荐）
- `http://127.0.0.1:8000/student`（单学生：时间轴散点/雷达对比/难度阶梯 动态展示）

注意：
- WebApp 只负责加载离线训练好的 `Pipeline` 并推理；请先运行：
  - `python 02_build_features.py`
  - `python 03_train_eval.py`（会保存 `Models/pipeline_logreg.joblib`）

---

## 端到端详解（从原始数据到推荐结果）

这部分给出一个“从 0 到 1”的闭环说明：**原始题库/日志 → 统一成 CleanData 标准表 → 校验 → 画像 → 特征 → 模型训练评估与选型 → 推荐生成与离线评估 → Web 展示**。所有步骤都对应仓库里的脚本与固定产物路径，确保可复现。

### 0）任务定义与离线评估口径

1) **预测任务（Pass Prediction）**  
把每条提交作为二分类样本：输入为本次提交的上下文特征 `x(u,p,t)`，输出 `P(AC)`。  

2) **推荐任务（Top‑K Recommendation）**  
对每个用户，从候选题中按 `P(AC)` 排序（可叠加规则），输出 Top‑K。  

3) **严格无泄漏的时间切分**  
离线评估按 `submission_id` 做时间切分：前 80% 为训练窗、后 20% 为测试窗。训练窗可用于训练模型与统计画像；测试窗只用于评估（ground truth），避免“看未来”。

---

### 1）原始题库 → problems.csv（清洗/转换/标注）

主流水线读取 `CleanData/problems.csv`。如果你从更原始的题库开始，推荐按下列链路把题库规范化到 problems 的字段约定（title/description/sample/time_limit/memory_limit/difficulty/tags 等）。

#### 1.1 题库 HTML → CSV（离线解析）

- 目的：把题面文本字段结构化，得到可用于标注与合并的 CSV。
- 脚本：`python Utils/tk_html_to_csv.py`
- 输入：题库 HTML 文件目录（例如把 `OriginalData/题库.zip` 解压后得到的 `.html` 文件夹）
- 输出：默认 `tk_problems.csv`（可通过 `--output` 指定路径）

示例：

```bash
# （示例）解压题库压缩包（如有）
# unzip OriginalData/题库.zip -d OriginalData

# 解析 HTML -> CSV
python Utils/tk_html_to_csv.py --input-dir OriginalData/TK题库 --output OriginalData/tk_problems.csv
```

#### 1.2 difficulty/tags 标注（可选：LLM 批量标注）

如果你的题库没有 `difficulty/tags`，可以用“题面 → 标注请求 → 批量标注 → 合并回 problems”这条链路：

1) CSV → prompts 文件（离线，不发网络请求；带 BEGIN/END 标记）：  
`python Utils/csv_to_requests.py --input OriginalData/tk_problems.csv --output OriginalData/请求.txt --output-format prompt`

2) 批量标注（会发网络请求；需 API Key；请勿提交密钥）：  
`python Utils/batch_label_qwen.py --prompts OriginalData/请求.txt --csv OriginalData/tk_problems.csv --output-csv OriginalData/tk_problems_labeled.csv`

3) 合并标注结果到 problems（并可输出校验报告）：  
`python Utils/merge_labels_into_originaldata_problems.py --labeled OriginalData/tk_problems_labeled.csv --problems CleanData/problems.csv --match order --inplace --validate-report Reports/validate/validate_report.txt`

4) tags 字段规范化（统一为 JSON 数组字符串 & 白名单过滤）：  
`python Utils/normalize_problems_tags_json.py --problems CleanData/problems.csv --tags CleanData/tags.csv --inplace`

说明：
- `CleanData/tags.csv(tag_name)` 定义标签白名单；标注/合并阶段会以白名单约束 tags。
- `problems.tags` 推荐保存为 JSON 数组字符串（例如 `["dp","graph"]`），仓库脚本对逗号分隔也做了兼容，但规范化后更稳。

---

### 2）提交日志准备：真实数据 or 模拟生产

主流水线需要 `CleanData/submissions.csv`。你可以直接导入真实 OJ 提交日志（并对齐字段约定），也可以用仓库内置脚本生成模拟提交。

#### 2.1 最小可运行的模拟提交（快速跑通流水线）

脚本：`python Utils/generate_originaldata_sim.py`

定位：
- 在你已经具备 `CleanData/problems.csv` 的前提下，生成最小可运行的 `students/languages/verdicts/submissions` 等表，让 `01~05` 能跑通。

核心模拟逻辑（对应脚本实现，简化描述）：
- 为每个用户采样能力 `ability ~ Beta(2,2)` 与坚持度 `perseverance ~ Beta(2,2)`
- 维护 `attempt_no(user,problem)` 与已 AC 的 solved 集，避免 AC 后继续提交同题
- 通过概率（学习效应 + 难度差）：  
  `p_ac = sigmoid((ability - diff01) * 4.0 + (attempt_no - 1) * 0.7)`  
  其中 `diff01` 为难度归一化到 [0,1]
- 若未 AC，则从 `{WA,TLE,RE,CE}` 按固定权重采样 verdict

#### 2.2 多因素相关的模拟提交生成（更贴近真实 OJ）

脚本：`python Utils/fill_exec_mem.py generate ...`

它会显式模拟多因素对 AC 的影响，并生成更“像真实数据”的相关性（难度、标签偏好、语言偏好、尝试次数学习效应、用户活跃度长尾等），并同时生成 `exec_time_ms/mem_kb`：

核心公式（对应 `Utils/fill_exec_mem.py`）：
- 设 `diff = difficulty / 10.0`
- `tmatch`：题目 tags 与用户 tag 偏好的匹配度
- `lmatch`：本次语言与用户语言偏好的匹配度
- `k`：同一题的第 k 次尝试（学习效应项）
- 加噪声 `eps ~ Uniform(-noise, noise)`
- 通过概率：
  `p_ac = sigmoid(a*level - b*diff + c*tmatch + d*lmatch + e*(k-1) + bias + eps)`
- 同一题是否继续尝试由 `perseverance` 与尝试次数衰减共同决定（脚本内置）

---

### 3）数据一致性校验（门禁）

脚本：`python Utils/validate_originaldata.py`

它会检查：
- submissions 外键是否能在 students/problems/languages/verdicts 中找到
- `ac` 与 `verdict=="AC"` 是否等价
- `attempt_no` 是否为正整数且对同一 `(user_id, problem_id)` 严格递增
- problems 的 `difficulty` 范围与 `tags` 格式/白名单一致性

建议输出报告文件（便于存档与排错）：

```bash
python Utils/validate_originaldata.py --report Reports/validate/validate_report.txt
```

---

### 4）画像派生（用户层概览）

脚本：`python 01_derive_students.py` → `CleanData/students_derived.csv`

输出字段（0~1 归一化）：
- `level`：用“已解题(AC) × 题目难度”的加权完成度表示能力
- `perseverance`：用“平均每题尝试次数”的对数归一化表示坚持度
- `lang_pref` / `tag_pref`：历史语言/标签偏好分布（JSON 字符串）

说明：
- 该画像主要用于 EDA/展示与“泄漏口径对比”（`Utils/compare_strict_vs_leaky.py`），并不是严格训练特征的来源。
- 严格训练特征在 `02_build_features.py` 内按时间顺序逐提交构造（只用“过去”历史），避免时间泄漏。

---

### 5）训练样本构造（严格无泄漏特征）

脚本：`python 02_build_features.py` → `FeatureData/train_samples.csv`

关键点（严格无泄漏）：
- 一行对应一次提交（以 `submission_id` 近似时间顺序）
- 对于每条 submission，只能使用该 submission **之前** 的历史统计构造特征
- `level/perseverance/lang_match/tag_match` 都是“滚动统计”的动态特征（不是全量聚合画像）
- 语言 one‑hot 与标签 multi‑hot 的列顺序由 `CleanData/languages.csv` / `CleanData/tags.csv` 固定，保证训练与推理列对齐

---

### 6）训练评估与模型选型

脚本：`python 03_train_eval.py`

做了什么：
- 按 `submission_id` 时间切分：前 80% 训练、后 20% 测试（更接近线上“用过去预测未来”）
- 训练多个候选模型：Logistic Regression / Linear SVM / Decision Tree
- 输出 `Models/metrics.csv`（Accuracy/Precision/Recall/F1）用于对比选型
- 保存可供 Web 推理的最终模型：`Models/pipeline_logreg.joblib`
- 输出混淆矩阵与对比图到 `Reports/fig/`

为何最终默认使用 Logistic Regression：
- 能直接输出概率 `P(AC)`，推荐排序可直接复用
- 离线指标通常与线性 SVM 接近，但概率可解释且便于做“成长带”过滤与校准分析

可选：严格 vs 泄漏口径对比（解释“指标虚高”）  
`python Utils/compare_strict_vs_leaky.py` → `Reports/compare/` + `Reports/fig/`

---

### 7）Top‑K 推荐生成与离线评估（严格无泄漏）

脚本：`python 04_recommend_eval.py`

核心流程：
- 训练一个 `P(AC)` 模型（默认 logistic regression）
- 对每个用户构造候选集：历史未 AC 的题目
- 给候选集打分并排序生成 Top‑K
- 默认启用“成长型推荐”：优先选择 `P(AC)` 位于 `[0.4, 0.7]` 的题（可调整）
- 离线评估：用测试时间窗内的 AC 作为 ground truth，计算 Hit@K / Precision@K 等

产物：
- `Reports/reco/recommendations_topk.csv`
- `Reports/reco/reco_metrics.csv`
- `Reports/fig/fig_hitk_curve.png` 等

---

### 8）EDA 与 sanity check（分布合理性）

脚本：`python 05_make_eda_plots.py` → `Reports/fig/fig_*.png`

这一步的定位是“快速发现数据异常/分布不合理”，例如：
- 难度与 AC 率是否大体负相关
- 尝试次数与 AC 率是否大体正相关
- 语言/标签分布是否极端偏斜

---

## 数据表结构（最小字段约定）

本项目的脚本尽量做了“宽松解析”（例如 `tags` 支持 JSON 数组或逗号分隔），但为了可复现与一致性，建议遵循以下字段约定。

### `CleanData/problems.csv`

必须字段：
- `problem_id`：题目唯一 ID（整数/可转整数）
- `difficulty`：题目难度（1–10，允许为空；会用中位数填充为 `difficulty_filled`）
- `tags`：题目标签（建议为 JSON 数组字符串，如 `["dp","greedy_sorting"]`；也可逗号分隔）

可选字段（用于 Web 展示更友好）：
- `title`、`description`、`sample_input`、`sample_output`、`hint`、`source`、`time_limit`、`memory_limit`

### `CleanData/submissions.csv`

必须字段（校验脚本会检查一致性）：
- `submission_id`：提交唯一 ID（本项目用它近似“时间顺序”做时间切分）
- `user_id`：用户 ID（需要能在 `students.csv` 里找到）
- `problem_id`：题目 ID（需要能在 `problems.csv` 里找到）
- `attempt_no`：同一 `user_id + problem_id` 的第几次尝试（正整数且应递增）
- `language`：语言名称（需要能在 `languages.csv` 的 `name` 列中找到）
- `verdict`：评测结果（需要能在 `verdicts.csv` 的 `name` 列中找到）
- `ac`：是否 AC（0/1，且要求与 `verdict=="AC"` 等价）

### `CleanData/tags.csv`
- `tag_name`：标签白名单（推荐系统的“题型空间”定义）

### `CleanData/languages.csv`
- `name`：语言白名单

---

## 方法概览（画像 → 特征 → 模型 → 推荐）

### 学生画像（`01_derive_students.py`）

输出 `CleanData/students_derived.csv`，核心字段：
- `level`：能力（0–1），按“做对题目的难度加权表现”归一化
- `perseverance`：坚持度（0–1），按“人均每题尝试次数”的对数归一化
- `lang_pref`：语言偏好分布（JSON 字典，键为语言名、值为占比）
- `tag_pref`：标签偏好分布（JSON 字典，键为标签、值为占比）

### 训练样本（`02_build_features.py`）

输出 `FeatureData/train_samples.csv`，一行一条提交：
- `y`：`ac`（0/1）
- `X`：示例特征包括
  - 数值：`attempt_no / difficulty_filled / level / perseverance / lang_match / tag_match`
  - One‑Hot：`lang_*`（语言）
  - Multi‑Hot：`tag_*`（题目标签，按 `tags.csv` 词表展开）

其中：
- `lang_match`：该次提交语言与用户历史语言偏好的匹配程度
- `tag_match`：题目标签与用户历史标签偏好的匹配程度
- 画像/偏好类特征按“该提交之前的历史”计算，避免时间泄漏

### 模型训练与评估（`03_train_eval.py`）

采用按 `submission_id` 排序后的时间切分（默认 80%/20%）：
- `logreg`：Logistic Regression（带 `StandardScaler(with_mean=False)`）
- `tree`：Decision Tree
- `svm_linear`：Linear SVM

产物：
- `Models/metrics.csv`：Accuracy/Precision/Recall/F1
- `Reports/fig/fig_cm_*.png`：各模型混淆矩阵
- `Reports/fig/fig_model_f1_compare.png`：F1 对比图

### Top‑K 推荐与评估（`04_recommend_eval.py`）

推荐思路：
- 先训练一个 `P(AC)` 模型（Logistic Regression）
- 对每个用户，从“历史未 AC 的题目”中为候选集打分并排序
- 默认启用“成长型推荐”：优先推荐 `P(AC)` 在 `[0.4, 0.7]` 的题目（可在脚本常量中调整）

评估口径（可复现、易写报告）：
- 按 `submission_id` 做训练/测试时间窗切分
- `Hit@K`：推荐的 Top‑K 中是否存在题目在测试窗口内被该用户 AC
- `Precision@K`：Top‑K 中命中 AC 的比例

产物：
- `Reports/reco/recommendations_topk.csv`：每个用户 Top‑K 推荐列表（含 `p_ac / difficulty / in_growth_band`）
- `Reports/reco/reco_metrics.csv`：Hit@K/Precision@K（含全量用户与“活跃用户”两套口径）
- `Reports/fig/fig_hitk_curve.png`：Hit@K 曲线
- `Reports/fig/fig_reco_coverage.png`：覆盖率/集中度（是否只推荐少数热门题）
- `Reports/fig/fig_reco_difficulty_hist.png`：单用户案例的推荐难度分布

---

## 工具脚本（Utils）

数据一致性/格式化：
- `Utils/validate_originaldata.py`：校验 `CleanData/*.csv` 的外键与关键字段一致性（并可输出报告）
- `Utils/normalize_problems_tags_json.py`：把 `problems.tags` 归一化为 JSON 数组字符串

题库解析与标注（可选，涉及网络时需自行配置）：
- `Utils/tk_html_to_csv.py`：解析题库 HTML 目录为 `tk_problems.csv`
- `Utils/csv_to_requests.py`：把题目信息转成带 BEGIN/END 标记的提示词文件（供批量标注）
- `Utils/batch_label_qwen.py`：调用 DashScope/Qwen 批量打 `difficulty/tags`（需要 `DASHSCOPE_API_KEY`，会产生网络请求）
- `Utils/merge_labels_into_originaldata_problems.py`：把标注结果合并回 `CleanData/problems.csv`

模拟数据（无真实数据时用于跑通全链路）：
- `Utils/generate_originaldata_sim.py`：生成一套最小可运行的 `CleanData/*`（学生/提交/语言/判题结果等）
- `Utils/fill_exec_mem.py`：为 submissions 填充/生成 `exec_time_ms/mem_kb`（或生成更强相关的模拟提交日志）

---

## 常见问题（Troubleshooting）

1) `train_samples.csv 缺少列 ...`  
先确认你运行了 `python 02_build_features.py`，并且 `CleanData/tags.csv`、`CleanData/languages.csv` 与数据一致。

2) `problems.tags` 是逗号分隔还是 JSON？  
推荐 JSON 数组字符串；若是逗号分隔，校验时加 `--accept-csv-tags`，或使用 `python Utils/normalize_problems_tags_json.py --inplace` 统一格式。

3) matplotlib 中文乱码/无法显示中文  
脚本里已设置常见中文字体回退；若你的系统缺字体，可安装任意中文字体（macOS/Windows 通常无需额外操作）。

---

## 许可证

见 `LICENSE`。
