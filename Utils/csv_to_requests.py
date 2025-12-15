import argparse
import csv
import json


PROMPT_TEMPLATE = """你是 OJ 题目标签与难度评估器。请先完整阅读题目信息，然后根据题目核心解法与算法特征完成标注。

【可选标签列表】（只能从中选择，不可自造，不可变形）：
array_string, hash_map, two_pointers_sliding, stack_queue, linked_list, tree, graph, dfs_bfs, dp, greedy_sorting, binary_search, math_bit

【标注规则】：
1. 标签按“主导解法/核心算法”选择
2. 只能选择 1–2 个标签
3. 难度为 1–10 的整数（OJ 常规算法题难度）
4. 不得输出任何解释性文字

【题目信息】：
标题：{title}
题目描述：{description}
样例输入：{sample_input}
样例输出：{sample_output}
提示：{hint}
来源：{source}
时间限制：{time_limit}
内存限制：{memory_limit}

【输出要求（极其重要）】：
- 只允许输出一行 JSON
- 只能包含两个键：difficulty, tags
- 不允许出现任何多余文字、解释、换行、空格或标点

输出格式示例（仅示例，不可照抄）：
{{"difficulty":3,"tags":["array_string"]}}"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert problems CSV into JSONL requests for /v1/chat/completions."
    )
    parser.add_argument("--input", default="tk_problems.csv", help="Input CSV path")
    parser.add_argument("--output", default="请求.txt", help="Output JSONL path")
    parser.add_argument("--model", default="qwen3-max", help="Model name")
    parser.add_argument(
        "--output-format",
        choices=("prompt", "jsonl"),
        default="prompt",
        help="Output format: plain prompt blocks or JSONL request objects",
    )
    parser.add_argument(
        "--begin-marker-template",
        default="<<<BEGIN:{id}>>>",
        help="Prompt mode only: line template before each prompt (supports {id} and {i})",
    )
    parser.add_argument(
        "--end-marker-template",
        default="<<<END:{id}>>>",
        help="Prompt mode only: line template after each prompt (supports {id} and {i})",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f_in, open(
        args.output, "w", encoding="utf-8", newline="\n"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for idx, row in enumerate(reader, start=1):
            request_id = f"request-{idx}"
            prompt = PROMPT_TEMPLATE.format(
                title=(row.get("title") or "").strip(),
                description=(row.get("description") or "").strip(),
                sample_input=(row.get("sample_input") or "").strip(),
                sample_output=(row.get("sample_output") or "").strip(),
                hint=(row.get("hint") or "").strip(),
                source=(row.get("source") or "").strip(),
                time_limit=(row.get("time_limit") or "").strip(),
                memory_limit=(row.get("memory_limit") or "").strip(),
            )

            if args.output_format == "prompt":
                if idx > 1:
                    f_out.write("\n\n")
                f_out.write(
                    args.begin_marker_template.format(id=request_id, i=idx).rstrip("\n")
                )
                f_out.write("\n")
                f_out.write(prompt)
                f_out.write("\n")
                f_out.write(
                    args.end_marker_template.format(id=request_id, i=idx).rstrip("\n")
                )
            else:
                request_obj = {
                    "custom_id": request_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    },
                }
                f_out.write(
                    json.dumps(request_obj, ensure_ascii=False, separators=(",", ":"))
                )
                f_out.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
