import json, re, math
import pandas as pd, numpy as np

PROBLEMS = "CleanData/problems.csv"  # adjust if needed
SUBS = "CleanData/submissions.csv"     # adjust if needed
TAGS = "CleanData/tags.csv"
LANGS = "CleanData/languages.csv"
OUT = "CleanData/students_derived.csv"

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

problems = pd.read_csv(PROBLEMS)
subs = pd.read_csv(SUBS)
tags = pd.read_csv(TAGS)
langs = pd.read_csv(LANGS)

problems["difficulty"] = pd.to_numeric(problems["difficulty"], errors="coerce")
diff_median = int(np.nanmedian(problems["difficulty"])) if problems["difficulty"].notna().any() else 5
problems["difficulty_filled"] = problems["difficulty"].fillna(diff_median).astype(int)
problems["tags_list"] = problems["tags"].apply(parse_json_list)

tag_vocab = tags["tag_name"].astype(str).tolist()
tag_set = set(tag_vocab)
problems["tags_norm"] = problems["tags_list"].apply(lambda lst: [t for t in lst if t in tag_set][:2])

subs["ac"] = pd.to_numeric(subs["ac"], errors="coerce").fillna(0).astype(int)

# per user-problem
up = subs.groupby(["user_id","problem_id"], as_index=False).agg(
    n_attempts=("submission_id","count"),
    solved=("ac","max")
).merge(problems[["problem_id","difficulty_filled","tags_norm"]], on="problem_id", how="left")

up["difficulty_filled"] = up["difficulty_filled"].fillna(diff_median).astype(int)
up["diff_norm"] = up["difficulty_filled"]/10.0

# level
num = (up["solved"] * up["diff_norm"]).groupby(up["user_id"]).sum()
den = up["diff_norm"].groupby(up["user_id"]).sum()
level = (num / (den + 1e-9)).reset_index(name="level")

# perseverance
attempt_stats = up.groupby("user_id")["n_attempts"].mean().reset_index(name="avg_attempts_per_problem")
p95 = np.percentile(attempt_stats["avg_attempts_per_problem"], 95)
den = math.log1p(p95) if p95>0 else 1.0
attempt_stats["perseverance"] = attempt_stats["avg_attempts_per_problem"].apply(lambda x: min(1.0, math.log1p(x)/den if den>0 else 0.0))

# lang_pref
known_langs = set(langs["name"].astype(str))
lang_counts = subs.groupby(["user_id","language"]).size().reset_index(name="cnt")
lang_counts = lang_counts[lang_counts["language"].isin(known_langs)]
lang_tab = lang_counts.pivot_table(index="user_id", columns="language", values="cnt", fill_value=0)
lang_tab = lang_tab.div(lang_tab.sum(axis=1).replace(0,1), axis=0)
lang_keys = sorted(list(known_langs))
lang_pref = lang_tab.apply(lambda r: json.dumps({k: float(r.get(k,0.0)) for k in lang_keys if r.get(k,0.0)>0}, ensure_ascii=False), axis=1).reset_index()
lang_pref.columns = ["user_id","lang_pref"]

# tag_pref
tag_rows=[]
for _, r in up[["user_id","tags_norm"]].iterrows():
    for t in (r["tags_norm"] if isinstance(r["tags_norm"], list) else []):
        tag_rows.append((int(r["user_id"]), t))
tag_df = pd.DataFrame(tag_rows, columns=["user_id","tag"])
tag_counts = tag_df.groupby(["user_id","tag"]).size().reset_index(name="cnt")
tag_tab = tag_counts.pivot_table(index="user_id", columns="tag", values="cnt", fill_value=0).reindex(columns=tag_vocab, fill_value=0)
tag_tab = tag_tab.div(tag_tab.sum(axis=1).replace(0,1), axis=0)
tag_pref = tag_tab.apply(lambda r: json.dumps({k: float(r.get(k,0.0)) for k in tag_vocab if r.get(k,0.0)>0}, ensure_ascii=False), axis=1).reset_index()
tag_pref.columns = ["user_id","tag_pref"]

users = pd.DataFrame({"user_id": sorted(subs["user_id"].unique().tolist())})
out = users.merge(level, on="user_id", how="left").merge(attempt_stats[["user_id","perseverance"]], on="user_id", how="left")\
           .merge(lang_pref, on="user_id", how="left").merge(tag_pref, on="user_id", how="left")
out["level"]=out["level"].fillna(0).clip(0,1)
out["perseverance"]=out["perseverance"].fillna(0).clip(0,1)
out["lang_pref"]=out["lang_pref"].fillna("{}")
out["tag_pref"]=out["tag_pref"].fillna("{}")

out.to_csv(OUT, index=False, encoding="utf-8-sig")
print("Wrote", OUT, "rows=", len(out))
