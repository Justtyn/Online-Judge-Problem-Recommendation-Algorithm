# Validation Report

## Notes
- Column normalization: strip + lowercase + hyphen→underscore + whitespace→underscore.
- Primary keys: drop null/non-int; keep first on duplicates.
- problems.difficulty: coerce numeric; fill missing with 5; clip to [0,10]; round half-up to int.
- users.level/users.perseverance: coerce numeric; fill missing with 0.5; clip to [0,1] as float.
- users.lang_pref/users.tag_pref: if invalid JSON -> set {}; drop invalid keys/values; normalize to sum=1 when total>0.
- submissions.exec_time_ms/memory_kb: coerce numeric; fill missing with 0; negatives set to 0; int.
- submissions.language: strip; must match languages.name (case-insensitive) then canonicalized; empty -> delete.
- submissions.verdict: strip+upper; must match verdicts.name; intermediate verdicts (WT/JG/PENDING/RUNNING if present) -> delete.

## Table Summaries
### languages
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/languages.xlsx` / sheet `languages`
- Rows: 6 → 6 (deleted 0)

### problems
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/problems.xlsx` / sheet `problems`
- Rows: 4559 → 4559 (deleted 0)

### submissions
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/submissions.xlsx` / sheet `submissions`
- Rows: 4046652 → 4023660 (deleted 22992)
- Deletes by reason:
  - student_id_fk_missing: 22952
  - verdict_intermediate_state: 34
  - problem_id_fk_missing: 6
- Fixes by reason:
  - exec_time_ms_filled_default_0: 2198
  - memory_kb_filled_default_0: 2198
- Notes:
  - intermediate verdicts considered: ['JG', 'WT']
  - intermediate verdict deletions: {'JG': 22, 'WT': 12}

### tags
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/tags.xlsx` / sheet `tags`
- Rows: 100 → 100 (deleted 0)

### users
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/students.xlsx` / sheet `users`
- Rows: 27444 → 27444 (deleted 0)
- Notes:
  - users.lang_pref non-empty: 0/27444; fixed: 0
  - users.tag_pref non-empty: 0/27444; fixed: 0

### verdicts
- Input: `/Users/justyn/Desktop/课程文件/人工智能/代码/OnlineJudgeRecommend/CleanData/data/formatted_xlsx/verdicts.xlsx` / sheet `verdicts`
- Rows: 13 → 13 (deleted 0)

## Key Distributions
### verdict distribution (top 20)
verdict
AC      1597832
WA      1582696
TLE      265420
CE       198779
REG      164342
PE       138888
OE        44644
MLE       15325
REP       15297
OFNR        279
IFNR        158

### language distribution (top 20)
language
c          3070273
c++         600561
python3     339432
python        8453
java          4680
python2        261

### difficulty distribution (0-10)
difficulty
0      378
1     1300
2      901
3      754
4      506
5      329
6      163
7      114
8       59
9       34
10      21

### problems.tags count per problem (min/median/max)
min=0, median=1, max=3

### users.lang_pref non-empty ratio
0/27444

### users.tag_pref non-empty ratio
0/27444

