# Repository Guidelines

## 项目结构与数据流

- `OriginalData/`：输入 CSV（已清洗、已打标签），默认只读。
- `Utils/`：数据校验/预处理的脚本与小型 CLI。
- 规划输出目录（按需创建）：`CleanData/` → `FeatureData/` → `Models/` → `Reports/`。
- `todo.md`：从画像、训练样本、建模评估到 Top‑K 推荐的交付清单。

建议流水线：
1）校验 `OriginalData/` → 2）派生画像 `CleanData/` → 3）构造训练样本 `FeatureData/` → 4）训练评估 `Models/` → 5）输出图表与推荐结果 `Reports/`。

## 常用命令（开发/运行/验证）

本仓库以 Python 脚本为主（暂无统一构建系统）。

- 创建并激活虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- 校验原始数据一致性：`python Utils/validate_originaldata.py`
- 输出校验报告：`python Utils/validate_originaldata.py --report Reports/validate_report.txt`
- 查看脚本参数：`python Utils/<script>.py --help`

新增依赖请同步记录（例如 `requirements.txt`），并尽量让网络调用可选/可关闭。

## 代码风格与命名

- Python：4 空格缩进；函数/变量用 `snake_case`；优先使用类型标注（参考 `Utils/*.py`）。
- 保持可复现：固定随机种子、避免隐式全局状态，优先“纯函数 + 小 CLI”。
- 产物命名：`CleanData/*.csv`、`FeatureData/*.csv`、`Models/metrics.csv`、`Reports/fig_*.png`。

## 测试与验证

- 当前无正式测试套件；改动后至少：
  - 重跑 `python Utils/validate_originaldata.py`
  - 抽查行数、空值率、关键分布（图保存到 `Reports/`）
- 如新增测试，建议使用 `pytest`，放在 `tests/`，命名 `test_*.py`。

## 提交与 PR 规范

- Commit：简短、祈使句风格；中英文均可（历史记录混用）。
- PR：说明改动动机与影响；给出可复现命令（精确到 `python ...`）；涉及结果产物时列出路径（如 `Reports/fig_*.png`、`Models/metrics.csv`）。

## 安全与配置

- 不要提交密钥/API Key；本地配置用 `.env`（已在 `.gitignore` 中忽略）。
- 非必要不提交体积很大的生成文件；报告必须产物除外。

## Agent/沟通说明

- 默认使用中文沟通与输出；如需英文，请在需求中明确指出。
