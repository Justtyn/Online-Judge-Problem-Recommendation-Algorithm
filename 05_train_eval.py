import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def setup_cn_font() -> None:
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

DATA = "FeatureData/train_samples.csv"
OUT_METRICS = "Models/metrics.csv"
OUT_DIR = "Reports"

df = pd.read_csv(DATA)
y = df["ac"].astype(int).values
id_order = df["submission_id"].values
X = df.drop(columns=["ac", "submission_id", "user_id", "problem_id"])

# boolean -> int
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)

# time split by submission_id
order = np.argsort(id_order)
X = X.iloc[order].reset_index(drop=True)
y = y[order]
split = int(len(X)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

models = {
  "logreg": Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=200))]),
  "tree": DecisionTreeClassifier(max_depth=10, random_state=42),
  "svm_linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LinearSVC())]),
}

rows=[]
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    })

metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
Path(OUT_METRICS).parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
print(metrics)

# confusion matrices
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
for name, model in models.items():
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["未AC", "AC"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    title_map = {"logreg": "逻辑回归", "tree": "决策树", "svm_linear": "线性SVM"}
    ax.set_title(f"混淆矩阵：{title_map.get(name, name)}")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_confusion_{name}.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/fig_cm_{name}.png", dpi=200)
    if name == "svm_linear":
        fig.savefig(f"{OUT_DIR}/fig_cm_svm_or_knn.png", dpi=200)
    plt.close(fig)

# model compare
plt.figure()
plt.bar(metrics["model"], metrics["f1"])
plt.title("模型F1对比（时间切分）")
plt.xlabel("模型")
plt.ylabel("F1")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_model_f1_compare.png", dpi=200)
plt.close()
