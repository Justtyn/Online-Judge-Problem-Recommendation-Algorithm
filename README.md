# Online Judge Problem Recommendation Algorithm
This repository is used for the project code of the undergraduate final course report for an artificial intelligence course, and it is currently in the stage of learning and exploration, and will be used for my own graduation design in the future.

## Quickstart（本地复现实验）

1. 创建虚拟环境并安装依赖（按你的环境自行安装 `pandas / numpy / scikit-learn / matplotlib` 等）
   - `python -m venv .venv && source .venv/bin/activate`
2. 确认输入数据存在（本项目使用 `CleanData/*.csv` 作为输入）
   - 例如：`CleanData/submissions.csv`、`CleanData/problems.csv`、`CleanData/tags.csv`
3. 运行流水线脚本（对应 todo 的第 3–6 步）
   - 第 3 步：生成学生画像：`python 03_derive_students.py`
   - 第 4 步：生成训练样本：`python 04_build_features.py`
   - 第 5 步：训练与评估模型：`python 05_train_eval.py`
   - 第 6 步：Top‑K 推荐与评估（Hit@K 口径：测试时间窗内是否 AC）：`python 06_recommend_eval.py`

输出目录：
- `CleanData/`：派生表（如 `students_derived.csv`）
- `FeatureData/`：训练样本（`train_samples.csv`）
- `Models/`：模型指标（`metrics.csv`）
- `Reports/`：混淆矩阵、Hit@K 曲线、推荐结果等（`recommendations_topk.csv`、`reco_metrics.csv`、`fig_*.png`）
