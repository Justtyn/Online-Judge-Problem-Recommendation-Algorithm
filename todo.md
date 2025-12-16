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

---

## Backlog：丰富优化建议（先记 TODO，暂不实现）

### P0（优先级最高：可复现 & 可信度）

- [x] 固化依赖：新增 `requirements.txt`（或 `pyproject.toml`）并区分“核心/可选”依赖（当前 README 里为手动安装）
- [ ] 统一流水线 CLI：给 `03_derive_students.py`、`04_build_features.py`、`05_train_eval.py`、`06_recommend_eval.py`、`07_make_eda_plots.py` 增加 `argparse`（输入/输出路径、随机种子、切分比例、成长区间等）
- [ ] 统一 schema/口径：抽出 `CleanData/*.csv` 的字段规范（类型/范围/主外键）为集中定义（如 `Utils/schema.py`），并让校验与特征构造复用
- [ ] 防数据泄漏检查：为时间切分（按 `submission_id`）增加“特征仅依赖历史”的一致性检查/报告（尤其 `attempt_no`、用户历史统计类特征）
- [ ] 概率质量评估：在 `05_train_eval.py` 增加 `ROC-AUC/PR-AUC/Brier score/校准曲线`，并写入 `Models/metrics.csv`

### P1（推荐效果 & 评估完善）

- [ ] 推荐指标扩展：在 `06_recommend_eval.py` 增加 `Recall@K / MAP@K / NDCG@K`（保留现有 `Hit@K/Precision@K`）
- [ ] 候选集口径更细：区分“未做过”与“做过但未 AC”，分别推荐与评估；训练负样本按用户分层/下采样
- [ ] 成长型推荐自适应：把固定概率区间（如 `[0.4,0.7]`）改成按 `level` 动态调整，并输出不同 `level` 分桶的指标曲线
- [ ] 多样性/去同质化：加入 `intra-list diversity`（基于标签距离）、`novelty`（惩罚过热题），并用简单重排（MMR）提升覆盖与多样性
- [ ] 冷启动策略：新用户/新题缺画像或缺标签时提供可解释 fallback（热门/难度阶梯/相似标签），并在 Web 端明确展示

### P1（工程化 & 性能）

- [x] WebApp 复用离线模型：离线训练阶段保存 `Pipeline`（`joblib/pickle`），`WebApp/server.py` 只加载与推理，避免每次启动重新训练
- [ ] 向量化特征构造：替换 `apply(axis=1)` 为向量化/映射计算，提升 `04_build_features.py` 对大数据量的速度与内存表现
- [ ] 运行元信息沉淀：每次跑流水线输出 `Reports/run_manifest.json`（参数、输入行数、随机种子、cutoff、时间、git hash 等），便于写报告追溯

### P2（代码质量 & 可维护性）

- [ ] 抽公共函数：收敛重复的 `parse_json_list/parse_json_dict/setup_cn_font` 到 `Utils/common.py`，加类型标注
- [ ] 最小测试集：引入 `pytest`，补少量快测（schema 校验、tags 解析、候选过滤不含已 AC、metrics 维度/列存在等）
- [ ] 图表命名一致：统一 `fig_confusion_*.png` 与 `fig_cm_*.png` 的产物命名，避免报告引用混乱
- [ ] 数据目录说明落地：明确 `OriginalData/` 缺失原因（数据不随仓库提交）与获取方式；强化“一键生成模拟数据”入口对齐 `Utils/generate_originaldata_sim.py`
