# TODO（第 3 步 → 最终交付：推荐算法模型开发路线）

> 数据源在 `OriginalData/`（已清洗、已打标签）。从现在开始按 **3→4→5→6** 顺序推进，保证每步都有明确产物（CSV + 图 + 可复现实验脚本）。

---

## 0. 项目结构（先统一输出目录）

- [x] 新建输出目录（若不存在就创建）
  - `CleanData/`（派生画像等“干净表”）
  - `FeatureData/`（可直接训练的样本表）
  - `Models/`（模型与指标）
  - `Reports/`（图表、推荐结果、最终可引用素材）
- [ ] 明确关键主键与字段映射（写在代码常量/README 里）
  - 用户：`student_id`（或 submissions 中的 user 标识列）
  - 题目：`problem_id`
  - 提交结果：`verdict`（映射 `AC`/非 AC）
  - 语言：`language` 或 `language_id`
  - 标签：题目多标签字段（如 `tags` 或 `tag_*`）
  - 难度：`difficulty`（若无则用题目通过率/分段规则构造）

数据文件参考：
- `OriginalData/submissions.csv`
- `OriginalData/problems.csv`
- `OriginalData/students.csv`（原始学生表，后续会派生画像）
- `OriginalData/tags.csv`、`OriginalData/languages.csv`、`OriginalData/verdicts.csv`

---

## 3. 行为反推学生画像（`students_derived`）

目标：从 `submissions.csv` + `problems.csv` 派生学生画像四列：`level`, `perseverance`, `lang_pref`, `tag_pref`。

### 3.1 输出（必须）

- [x] 生成 `CleanData/students_derived.csv`
  - 最少字段：`student_id, level, perseverance, lang_pref, tag_pref`
  - 建议保留可解释的中间统计列（可选）：`attempts`, `unique_problems`, `ac_rate`, `avg_attempts_per_problem`

### 3.2 画像定义（建议口径，便于写报告）

- [ ] `level`（能力分层）：基于用户的 AC 表现 + 难度（例如：`mean(difficulty of solved)` 或 `AC率×题目难度加权`），再分箱成 3–5 档（如 L1–L5）
- [ ] `perseverance`（坚持度）：基于“同题重试次数/每题平均提交次数/放弃率”等构造，再分箱成 3–5 档
- [ ] `lang_pref`（语言偏好）：用户提交最多的语言
- [ ] `tag_pref`（题型偏好）：用户做题最多的标签（题目多标签时可按出现次数累计）

### 3.3 可视化（强烈建议，产物固定命名）

- [x] `Reports/fig_level_hist.png`：用户 `level` 分布直方图
- [x] `Reports/fig_perseverance_hist.png`：用户 `perseverance` 分布直方图
- [x] `Reports/fig_lang_dist.png`：语言总体分布柱状图（Python/C/C++/JS/JAVA/GO 等）
- [x] `Reports/fig_tag_dist.png`：标签总体分布柱状图（12 类题型占比）
- [x] `Reports/fig_user_activity.png`：用户做题量/提交量分布（长尾）

验收点：
- [ ] 分布符合常识：做题量长尾明显；难题更少；常用语言占比较高

---

## 4. 构建可训练样本表（`FeatureData/train_samples`）

目标：以 “一行一提交” 为粒度，联表生成 `sklearn` 可直接训练的数据集（`X + y`）。

### 4.1 输出（必须）

- [x] 生成 `FeatureData/train_samples.csv`
  - `y`：是否 AC（1/0）
  - `X`：建议包含
    - 用户画像：`level`, `perseverance`, `lang_pref`, `tag_pref`
    - 题目属性：`difficulty`, `tags`（多标签可 multi-hot 或拆 12 维）
    - 提交属性：`language`, `attempt_no`（同一用户-题目第几次尝试）
    - 历史统计（防泄漏：只用该提交前的历史）：用户历史 AC 率、近期表现等（可选加分）
- [ ] （可选）输出 `FeatureData/feature_spec.md`：特征解释与口径

### 4.2 可视化（强烈建议）

- [x] `Reports/fig_difficulty_vs_ac.png`：`difficulty` vs AC 率（难度越高通过率应下降）
- [x] `Reports/fig_attemptno_vs_ac.png`：`attempt_no` vs AC 率曲线
- [x] `Reports/fig_tag_acrate.png`：各标签平均 AC 率柱状图（题目多标签已展开）
- [x] `Reports/fig_lang_acrate.png`：各语言 AC 率柱状图

验收点：
- [ ] 样本表无空主键；`y` 分布不过度极端；可被 `pandas.read_csv` 直接读取训练

---

## 5. 时间切分训练与评估（`Models/`）

目标：按时间切分训练/测试（优先用 `timestamp`；没有就用 `submission_id` 近似时间），训练多模型对比并保存结果。

### 5.1 模型（建议 3 个，贴合课程）

- [x] Logistic Regression（主模型，输出可解释系数）
- [x] Decision Tree（对比，输出特征重要性）
- [x] SVM 或 KNN（二选一对比）

### 5.2 输出（必须）

- [x] `Models/metrics.csv`：每个模型的 `Accuracy/Precision/Recall/F1`（train/test 都建议记录）
- [ ] （可选）`Models/model_*.pkl`：保存训练好的模型/预处理器（pipeline）

### 5.3 可视化（强烈建议）

- [ ] `Reports/fig_metrics_train_test.png`：训练集 vs 测试集 指标对比柱状图（判断过拟合）
- [x] `Reports/fig_cm_logreg.png`、`Reports/fig_cm_tree.png`、`Reports/fig_cm_svm_or_knn.png`：混淆矩阵 heatmap
- [ ] `Reports/fig_roc_logreg.png`：ROC 曲线（至少 Logistic）
- [ ] `Reports/fig_coef_logreg.png` 或 `Reports/fig_importance_tree.png`：可解释性图（系数/重要性）

验收点：
- [ ] 时间切分无数据泄漏（不能把未来信息用到过去）
- [ ] 指标能稳定复现（固定随机种子、固定切分规则）

---

## 6. Top-K 推荐与推荐评估（`Reports/`）

目标：用预测概率做推荐；采用“成长型推荐”：偏好预测成功概率在 **0.4–0.7** 区间的题目（可配置）。

### 6.1 推荐产物（必须）

- [x] `Reports/recommendations_topk.csv`
  - 至少包含 20–50 个用户示例
  - 字段建议：`student_id, problem_id, score(p_ac), difficulty, tags, reason(optional)`

### 6.2 评估指标（必须）

- [x] 计算 `Hit@K` / `Precision@K`（K=1,3,5,10）
  - 评估集：时间切分后的测试窗口（用“后续是否 AC/是否尝试”做命中定义，按你报告口径固定）

### 6.3 可视化（建议）

- [x] `Reports/fig_hitk_curve.png`：Hit@K 随 K 变化曲线
- [x] `Reports/fig_reco_difficulty_hist.png`：用户案例推荐题难度分布（是否符合“成长型”）
- [x] `Reports/fig_reco_coverage.png`：覆盖率/集中度（避免只推荐少数热门题）

验收点：
- [x] 推荐只从“用户未 AC（或未做过）”题集中选
- [x] 概率区间过滤与 Top-K 逻辑可复现（写成函数/脚本）

---

## 最终交付清单（写报告用）

- [x] 所有 CSV：`CleanData/students_derived.csv`、`FeatureData/train_samples.csv`、`Models/metrics.csv`、`Reports/recommendations_topk.csv`
- [ ] 所有图：第 3–6 步图表齐全、命名固定、可直接插入报告
- [ ] 一键运行入口（建议）：`python -m ...` 或 `make all`（可选，但强烈推荐）
