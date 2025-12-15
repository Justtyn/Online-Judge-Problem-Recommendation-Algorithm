import json, re
import pandas as pd, numpy as np
from pathlib import Path

PROBLEMS = "CleanData/problems.csv"
SUBS = "CleanData/submissions_clean.csv"  # 兼容：也可能叫 submissions.csv
STUDENTS = "CleanData/students_derived.csv"
TAGS = "CleanData/tags.csv"
LANGS = "CleanData/languages.csv"
OUT = "FeatureData/train_samples.csv"


def first_existing_path(*candidates: str) -> str:
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"找不到输入文件，候选路径：{candidates!r}")

def parse_json_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return []
    try:
        v = json.loads(s)
        if isinstance(v, list): return [str(t) for t in v]
    except Exception:
        pass
    s = s.strip("[]")
    parts = re.split(r"[;,]\s*|\s+\|\s+|\s+,\s+", s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def parse_json_dict(x):
    if pd.isna(x): return {}
    if isinstance(x, dict): return x
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}

problems = pd.read_csv(PROBLEMS)
subs = pd.read_csv(first_existing_path(SUBS, "CleanData/submissions.csv"))
students = pd.read_csv(STUDENTS)
tags = pd.read_csv(TAGS)
langs = pd.read_csv(LANGS)

problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
problems["tags_norm"] = problems["tags"].apply(parse_json_list)

tag_vocab = tags["tag_name"].astype(str).tolist()
tag_set=set(tag_vocab)
problems["tags_norm"] = problems["tags_norm"].apply(lambda lst: [t for t in lst if t in tag_set][:2])

subs["ac"]=pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

df = subs.merge(problems[["problem_id","difficulty_filled","tags_norm"]], on="problem_id", how="left")\
        .merge(students[["user_id","level","perseverance","lang_pref","tag_pref"]], on="user_id", how="left")

students["user_id"] = pd.to_numeric(students["user_id"], errors="coerce")
students_lang_pref = students.dropna(subset=["user_id"]).set_index("user_id")["lang_pref"].apply(parse_json_dict).to_dict()
students_tag_pref = students.dropna(subset=["user_id"]).set_index("user_id")["tag_pref"].apply(parse_json_dict).to_dict()

def tag_match(row):
    pref = students_tag_pref.get(float(row["user_id"]), {}) or students_tag_pref.get(int(row["user_id"]), {})
    tags_list = row["tags_norm"] if isinstance(row["tags_norm"], list) else []
    if not tags_list: return 0.0
    vals = [float(pref.get(t, 0.0)) for t in tags_list]
    return float(sum(vals))/len(vals)

def lang_match(row):
    pref = students_lang_pref.get(float(row["user_id"]), {}) or students_lang_pref.get(int(row["user_id"]), {})
    return float(pref.get(str(row["language"]), 0.0))

known_langs = sorted(list(set(langs["name"].astype(str))))
lang_ohe = pd.get_dummies(df["language"], prefix="lang")
for l in known_langs:
    c=f"lang_{l}"
    if c not in lang_ohe.columns: lang_ohe[c]=0
lang_ohe = lang_ohe[[f"lang_{l}" for l in known_langs]]

def multihot(lst, vocab):
    s=set(lst) if isinstance(lst, list) else set()
    return [1 if t in s else 0 for t in vocab]

tag_mh = pd.DataFrame([multihot(t, tag_vocab) for t in df["tags_norm"]], columns=[f"tag_{t}" for t in tag_vocab])

out = pd.concat([
    df[["submission_id","user_id","problem_id","attempt_no"]].astype(int),
    df[["difficulty_filled","level","perseverance"]].fillna(0),
    pd.DataFrame({
        "lang_match": df.apply(lang_match, axis=1).astype(float),
        "tag_match": df.apply(tag_match, axis=1).astype(float),
    }).fillna(0.0),
    lang_ohe,
    tag_mh,
    df[["ac"]].astype(int)
], axis=1)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print("Wrote", OUT, "rows=", len(out), "cols=", out.shape[1])
