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

DATA = "FeatureData/train_samples.csv"
OUT_METRICS = "Models/metrics.csv"
OUT_DIR = "Reports"

df = pd.read_csv(DATA)
y = df["ac"].astype(int).values
X = df.drop(columns=["ac"])

# boolean -> int
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)

# time split by submission_id
order = np.argsort(df["submission_id"].values)
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion matrix: {name}")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_confusion_{name}.png", dpi=200)
    plt.close(fig)

# model compare
plt.figure()
plt.bar(metrics["model"], metrics["f1"])
plt.title("Model F1 comparison (time split)")
plt.xlabel("model")
plt.ylabel("F1")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_model_f1_compare.png", dpi=200)
plt.close()
