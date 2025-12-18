# Online Judge 题目推荐算法（OJ Problem Recommendation）

本仓库实现了一个面向 Online Judge 的“题目推荐算法”端到端流水线：从行为日志反推学生画像、构造训练样本、训练/评估 AC 预测模型，到基于预测概率的 Top‑K 推荐与推荐评估，并提供本地 Web 页面用于展示图表与交互式自定义推荐。

- 任务定义：把“某次提交是否 AC”建模为二分类问题，输出 `P(AC)`；推荐阶段用 `P(AC)` 作为排序分数（并支持“成长型推荐”概率区间过滤）

---

## 目录结构（数据流）

> 建议流水线：校验数据 → 派生画像 → 构造样本 → 训练评估 → Top‑K 推荐评估 → 产出图表与报告素材

```
.
├── CleanData/        # 输入/派生的干净表（本仓库默认从这里读取）
├── FeatureData/      # 可直接训练的样本表（X + y）
├── Models/           # 模型评估指标（可扩展保存模型文件）
├── Reports/          # 图表、推荐结果、评估指标（写报告直接引用）
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
- 如果你没有真实数据，可用 `Utils/generate_originaldata_sim.py` 生成一套可跑通流水线的模拟数据（见下文）。

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
python Utils/validate_originaldata.py --report Reports/validate_report.txt
```

### 2.5）诊断/校验：为什么推荐偏“全是难度 1”

当你发现某些学生提交了不少中等题，但 Top‑K 却被低难度“刷屏”时，可用诊断脚本输出量化证据（含候选集按难度的 `P(AC)` 分布、Top‑K 列表、以及可失败的校验 flags）：

```bash
# 单个学生：输出文本报告 + 图表到 Reports/
python Utils/diagnose_reco_bias.py --user-id 104 --cutoff-pct 0.50 --min-p 0.40 --max-p 0.70 --k 10 --plot

# 单个学生：命中指定 flags 则退出码=2（便于接 CI/批处理）
python Utils/diagnose_reco_bias.py --user-id 104 --fail-on easy_bias_vs_history,score_plateau_topk

# 批量扫描（默认扫所有有 submissions 的用户）：输出 CSV；可用 --fail-if-any 作为“全局校验”
python Utils/diagnose_reco_bias.py --scan --max-users 300 --fail-if-any
```

默认产物：
- `Reports/diag_user_<user_id>.txt`
- `Reports/fig_diag_user_<user_id>_reco_diff_hist.png`
- `Reports/fig_diag_user_<user_id>_candidate_p_by_diff.png`
- `Reports/diag_scan_users.csv`（scan 模式）

### 3）运行推荐算法流水线（03 → 06）

说明：以下脚本默认以仓库根目录为基准解析 `CleanData/FeatureData/Models/Reports` 路径，因此可以在任意工作目录运行。

```bash
python 01_derive_students.py      # 画像：CleanData/students_derived.csv
python 02_build_features.py       # 样本：FeatureData/train_samples.csv
python 03_train_eval.py           # 模型：Models/metrics.csv + Reports/fig_cm_*.png
python 04_recommend_eval.py       # 推荐：Reports/recommendations_topk.csv + Reports/reco_metrics.csv
python 05_make_eda_plots.py       # EDA：Reports/fig_*.png（用于写报告的“分布合理性”图）
```

关键产物（路径固定，便于引用）：
- `CleanData/students_derived.csv`：学生画像（`level / perseverance / lang_pref / tag_pref`）
- `FeatureData/train_samples.csv`：训练样本（可直接喂给 sklearn）
- `Models/metrics.csv`：分类模型对比指标（Accuracy/Precision/Recall/F1）
- `Reports/recommendations_topk.csv`：Top‑K 推荐结果
- `Reports/reco_metrics.csv`：Hit@K / Precision@K 等推荐评估指标
- `Reports/fig_*.png`：所有图表（EDA + 混淆矩阵 + Hit@K 曲线等）

---

## 本地 Web（图表展示 + 自定义推荐）

Web 页面会：
1) 自动展示 `Reports/` 下所有 `fig_*.png`（并按“数据层/训练层/推荐层”分区）  
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
- `Reports/fig_cm_*.png`：各模型混淆矩阵
- `Reports/fig_model_f1_compare.png`：F1 对比图

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
- `Reports/recommendations_topk.csv`：每个用户 Top‑K 推荐列表（含 `p_ac / difficulty / in_growth_band`）
- `Reports/reco_metrics.csv`：Hit@K/Precision@K（含全量用户与“活跃用户”两套口径）
- `Reports/fig_hitk_curve.png`：Hit@K 曲线
- `Reports/fig_reco_coverage.png`：覆盖率/集中度（是否只推荐少数热门题）
- `Reports/fig_reco_difficulty_hist.png`：单用户案例的推荐难度分布

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
