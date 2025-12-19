"""
03_train_eval.py

用途
- 读取 `FeatureData/train_samples.csv` 训练二分类模型（是否 AC）。
- 使用按 `submission_id` 的时间切分做离线评估（避免随机切分带来的信息泄漏风险）。
- 输出评估指标 `Models/metrics.csv`、保存可供 WebApp 推理的 Pipeline、
  并在 `Reports/fig/` 下生成混淆矩阵与模型对比图。

说明
- 本脚本默认以“脚本”方式运行；为了可读性/可复用性，主体逻辑封装在 `main()`。
- 为了让 matplotlib 在无 GUI 环境可用，强制使用 Agg 后端，并把缓存目录放到仓库 `.cache/`。
"""

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# ---- 运行环境准备：把缓存/配置写入仓库内，避免污染用户目录 & 便于 CI/容器 ----
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def setup_cn_font() -> None:
    """配置 matplotlib 中文字体回退，保证图表标题/标签可显示中文。"""
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

DATA = ROOT / "FeatureData/train_samples.csv"
OUT_METRICS = ROOT / "Models/metrics.csv"
OUT_PIPELINE = ROOT / "Models/pipeline_logreg.joblib"
# 图表统一收敛到 Reports/fig/
OUT_DIR = ROOT / "Reports/fig"
RANDOM_SEED = 42

def load_train_samples(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    读取训练样本并拆分出特征与标签。

    返回
    - X: 仅包含特征列的 DataFrame
    - y: 标签（0/1）
    - submission_id: 用于时间切分的排序键
    - feature_cols: 特征列名（用于推理时对齐）
    """
    if not data_path.exists():
        raise FileNotFoundError(f"找不到训练样本：{data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"ac", "submission_id", "user_id", "problem_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"训练样本缺少必要列：{sorted(missing)}")

    # y: 是否 AC
    y = df["ac"].astype(int).to_numpy()
    # submission_id: 用于按“提交时间”近似排序（该字段在上游构造时应随时间递增）
    submission_id = df["submission_id"].to_numpy()

    # 特征列：排除标签与 ID 类字段（避免直接泄漏）
    X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"])
    feature_cols = list(X.columns)

    # 统一把 bool 特征转成 0/1，便于 scaler 与模型处理
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X, y, submission_id, feature_cols


def time_split_by_submission_id(
    X: pd.DataFrame,
    y: np.ndarray,
    submission_id: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    按 submission_id 做时间切分（前 80% 训练，后 20% 测试）。

    这样可以更接近线上分布漂移/时间演化场景，避免随机切分造成“未来信息”泄漏。
    """
    if len(X) != len(y) or len(X) != len(submission_id):
        raise ValueError("X/y/submission_id 行数不一致")

    order = np.argsort(submission_id)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y[order]

    split_idx = int(len(X_sorted) * train_ratio)
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]
    return X_train, X_test, y_train, y_test


def build_candidate_models(random_seed: int) -> dict[str, Any]:
    """构建候选模型集合（用于对比离线指标）。"""
    # NOTE: `with_mean=False` 在稀疏输入时是必须的；当前特征可能较稀疏（0/1、one-hot 等），
    # 为兼容潜在的稀疏矩阵/避免不必要的 densify，统一使用 `with_mean=False`。
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, random_state=random_seed)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=10, random_state=random_seed),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LinearSVC(random_state=random_seed)),
            ]
        ),
    }


def eval_models_and_collect_metrics(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """训练各模型并在测试集上计算 Accuracy/Precision/Recall/F1。"""
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def save_offline_pipeline_for_webapp(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    out_path: Path,
    random_seed: int,
) -> None:
    """用全量样本训练一个逻辑回归 Pipeline，并以 joblib 格式持久化供 WebApp 推理使用。"""
    final_logreg = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, random_state=random_seed)),
        ]
    )

    # 统一转成 float32：通常能减小模型序列化体积，也能让推理时 dtype 更稳定。
    final_logreg.fit(X.to_numpy(dtype=np.float32), y.astype(int))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_logreg,
            "feature_cols": feature_cols,
            "random_seed": random_seed,
            "train_rows": int(len(X)),
        },
        out_path,
    )


def plot_confusion_matrices(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    out_dir: Path,
) -> None:
    """为每个模型绘制并保存混淆矩阵。"""
    title_map = {"logreg": "逻辑回归", "tree": "决策树", "svm_linear": "线性SVM"}
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["未AC", "AC"])

        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"混淆矩阵：{title_map.get(name, name)}")
        plt.tight_layout()

        model_label = title_map.get(name, name)
        fig.savefig(out_dir / f"fig_混淆矩阵_{model_label}.png", dpi=200)

        # 兼容既有报告里对该文件名的引用（svm 与 knn 早期做过对比）。
        if name == "svm_linear":
            fig.savefig(out_dir / "fig_混淆矩阵_SVM或KNN.png", dpi=200)

        plt.close(fig)


def plot_model_f1_compare(metrics: pd.DataFrame, out_dir: Path) -> None:
    """绘制各模型 F1 对比柱状图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(metrics["model"], metrics["f1"])
    plt.title("模型F1对比（时间切分）")
    plt.xlabel("模型")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_模型F1对比.png", dpi=200)
    plt.close()


def main() -> None:
    # 1) 读取训练样本
    X, y, submission_id, feature_cols = load_train_samples(DATA)

    # 2) 时间切分：前 80% 训练，后 20% 测试
    X_train, X_test, y_train, y_test = time_split_by_submission_id(X, y, submission_id, train_ratio=0.8)

    # 3) 训练与评估：输出离线指标表
    models = build_candidate_models(random_seed=RANDOM_SEED)
    metrics = eval_models_and_collect_metrics(models, X_train, y_train, X_test, y_test)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(metrics)

    # 4) 保存离线 Pipeline（供 WebApp 直接加载与推理）
    save_offline_pipeline_for_webapp(
        X=X,
        y=y,
        feature_cols=feature_cols,
        out_path=OUT_PIPELINE,
        random_seed=RANDOM_SEED,
    )
    print("Saved pipeline to", OUT_PIPELINE)

    # 5) 产出可视化：混淆矩阵 + F1 对比图
    plot_confusion_matrices(models=models, X_test=X_test, y_test=y_test, out_dir=OUT_DIR)
    plot_model_f1_compare(metrics=metrics, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
