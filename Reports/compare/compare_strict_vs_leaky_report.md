# Strict（无泄漏） vs Leaky（泄漏）对比报告

- strict: `/Users/justyn/Documents/GitHub/Online-Judge-Problem-Recommendation-Algorithm/FeatureData/train_samples.csv`
- leaky: 由 `/Users/justyn/Documents/GitHub/Online-Judge-Problem-Recommendation-Algorithm/CleanData/submissions.csv` + `/Users/justyn/Documents/GitHub/Online-Judge-Problem-Recommendation-Algorithm/CleanData/students_derived.csv` 重建（等价于旧版特征口径）
- time split: 0.80（cutoff_submission_id=240000）

## 1) 关键特征分布（均值/中位数/P95/标准差）

| feature | variant | mean | p50 | p95 | std |
| --- | --- | --- | --- | --- | --- |
| difficulty_filled | strict | 4.74234 | 5 | 8 | 1.87568 |
| difficulty_filled | leaky | 4.74234 | 5 | 8 | 1.87568 |
| attempt_no | strict | 1.42502 | 1 | 3 | 0.742557 |
| attempt_no | leaky | 1.42502 | 1 | 3 | 0.742557 |
| level | strict | 0.651584 | 0.671756 | 0.92735 | 0.193482 |
| level | leaky | 0.657426 | 0.669733 | 0.910839 | 0.174877 |
| perseverance | strict | 0.822697 | 0.814522 | 1 | 0.113713 |
| perseverance | leaky | 0.828918 | 0.817795 | 1 | 0.100303 |
| lang_match | strict | 0.382614 | 0.363636 | 0.791045 | 0.237928 |
| lang_match | leaky | 0.385619 | 0.364286 | 0.785088 | 0.229416 |
| tag_match | strict | 0.140563 | 0.134518 | 0.25 | 0.0665085 |
| tag_match | leaky | 0.139854 | 0.135 | 0.243108 | 0.0583377 |

## 2) 同一用户画像随时间变化的量化（std 越大表示越动态）

| feature | variant | mean_user_std | p50_user_std | p95_user_std |
| --- | --- | --- | --- | --- |
| level | strict | 0.0775744 | 0.0753594 | 0.109544 |
| level | leaky | 0 | 0 | 0 |
| perseverance | strict | 0.0559907 | 0.0554152 | 0.0707635 |
| perseverance | leaky | 0 | 0 | 0 |

## 3) 模型效果对比（同一切分规则）

| variant | model | accuracy | precision | recall | f1 | roc_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaky | dummy_most_frequent | 0.50795 | 0 | 0 | 0 |  |  |
| strict | dummy_most_frequent | 0.50795 | 0 | 0 | 0 |  |  |
| leaky | dummy_stratified | 0.500683 | 0.492635 | 0.493886 | 0.49326 |  |  |
| strict | dummy_stratified | 0.500683 | 0.492635 | 0.493886 | 0.49326 |  |  |
| leaky | logreg | 0.686933 | 0.681005 | 0.68428 | 0.682638 | 0.754227 | 0.201274 |
| strict | logreg | 0.683217 | 0.681173 | 0.669614 | 0.675344 | 0.749705 | 0.203176 |
| leaky | svm_linear | 0.687167 | 0.681167 | 0.68472 | 0.682939 |  |  |
| strict | svm_linear | 0.68325 | 0.681445 | 0.669004 | 0.675167 |  |  |
| leaky | tree | 0.678783 | 0.675754 | 0.667446 | 0.671574 |  |  |
| strict | tree | 0.67425 | 0.671762 | 0.660908 | 0.666291 |  |  |

## 4) 推荐评估对比（Hit@K / Precision@K）

| variant | k | hit_at_k_all | precision_at_k_all | hit_at_k_active | precision_at_k_active | users_all | users_active | growth_band | cutoff_submission_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaky | 1 | 0.018018 | 0.018018 | 0.018018 | 0.018018 | 999 | 999 | [0.4,0.7] | 240000 |
| strict | 1 | 0.002002 | 0.002002 | 0.002002 | 0.002002 | 999 | 999 | [0.4,0.7] | 240000 |
| leaky | 3 | 0.039039 | 0.014014 | 0.039039 | 0.014014 | 999 | 999 | [0.4,0.7] | 240000 |
| strict | 3 | 0.016016 | 0.00533867 | 0.016016 | 0.00533867 | 999 | 999 | [0.4,0.7] | 240000 |
| leaky | 5 | 0.0530531 | 0.0122122 | 0.0530531 | 0.0122122 | 999 | 999 | [0.4,0.7] | 240000 |
| strict | 5 | 0.036036 | 0.00760761 | 0.036036 | 0.00760761 | 999 | 999 | [0.4,0.7] | 240000 |
| leaky | 10 | 0.0860861 | 0.0102102 | 0.0860861 | 0.0102102 | 999 | 999 | [0.4,0.7] | 240000 |
| strict | 10 | 0.0760761 | 0.00940941 | 0.0760761 | 0.00940941 | 999 | 999 | [0.4,0.7] | 240000 |

## 5) 概率质量诊断（ROC/PR/校准/Brier 分解）

| variant | roc_auc | ap | brier | reliability | resolution | uncertainty |
| --- | --- | --- | --- | --- | --- | --- |
| strict | 0.749705 | 0.736744 | 0.203176 | 0.000257513 | 0.0458952 | 0.249937 |
| leaky | 0.754227 | 0.741588 | 0.201274 | 6.73757e-05 | 0.0479833 | 0.249937 |

## 结论建议（如何判断“是否失真”）

- 如果 leaky 显著高于 strict，说明旧口径存在“看未来”的时间泄漏，评估被抬高。
- strict 指标更接近线上真实可用效果；推荐评估也应优先看 strict 版本。

