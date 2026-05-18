import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from .prompt import (
        PARSE_PROMPT_TASK_1,
        PARSE_PROMPT_TASK_2,
        PARSE_PROMPT_TASK_3,
    )
except ImportError:
    from prompt import (
        PARSE_PROMPT_TASK_1,
        PARSE_PROMPT_TASK_2,
        PARSE_PROMPT_TASK_3,
    )


VIDEO_PAIR_KEYS = (
    ("video_path_a", "video_path_b"),
    ("video_path_A", "video_path_B"),
    ("video_a_path", "video_b_path"),
    ("video_A_path", "video_B_path"),
    ("video_a", "video_b"),
    ("video_A", "video_B"),
    ("Video A", "Video B"),
)
PARSE_PROMPTS = {
    1: PARSE_PROMPT_TASK_1,
    2: PARSE_PROMPT_TASK_2,
    3: PARSE_PROMPT_TASK_3,
}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_video_paths(item):
    if "video_path" in item:
        video_path = item["video_path"]
        if isinstance(video_path, (list, tuple)):
            return list(video_path)
        return [video_path]

    for key_a, key_b in VIDEO_PAIR_KEYS:
        if key_a in item and key_b in item:
            return [item[key_a], item[key_b]]

    return []


def sample_key(item):
    if "sample_id" in item:
        return f"sample_id:{item['sample_id']}"

    video_paths = get_video_paths(item)
    if video_paths:
        return json.dumps(video_paths, ensure_ascii=False)
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def clean_response(response):
    response = response or ""
    response = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL)
    return response.strip()


def normalize_task1_answer(answer):
    answer = clean_response(answer).strip().lower().strip(".。,:; ")
    if answer in {"yes", "no"}:
        return answer

    final_match = re.search(r"(?:final answer|answer)\s*[:：]\s*(yes|no)\b", answer)
    if final_match:
        return final_match.group(1)

    matches = re.findall(r"\b(yes|no)\b", answer)
    return matches[-1] if matches else "Invalid"


def normalize_task2_answer(answer):
    answer = clean_response(answer)
    compact = answer.lower()

    if re.fullmatch(r"<?video\s*a>?", compact):
        return "<Video A>"
    if re.fullmatch(r"<?video\s*b>?", compact):
        return "<Video B>"

    matches = re.findall(r"<?video\s*([ab])>?", compact)
    if matches:
        return f"<Video {matches[-1].upper()}>"
    return "Invalid"


def valid_option_letters(item):
    options = item.get("option") or item.get("options")
    if isinstance(options, list) and options:
        return {chr(ord("A") + index) for index in range(len(options))}
    return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_task3_answer(answer, item):
    answer = clean_response(answer).upper()
    if answer == "INVALID":
        return "Invalid"

    valid_letters = valid_option_letters(item)
    letters = []
    for letter in re.findall(r"\b[A-Z]\b", answer):
        if letter in valid_letters and letter not in letters:
            letters.append(letter)

    return ",".join(letters) if letters else "Invalid"


def normalize_answer(answer, item):
    task_type = int(item["task_type"])
    if task_type == 1:
        return normalize_task1_answer(answer)
    if task_type == 2:
        return normalize_task2_answer(answer)
    if task_type == 3:
        return normalize_task3_answer(answer, item)
    raise ValueError(f"Unsupported task_type: {task_type}")


def normalize_gold_answer(answer, task_type):
    fake_item = {"task_type": task_type, "option": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    return normalize_answer(answer, fake_item)


def is_correct(item):
    if "answer" not in item:
        return None

    task_type = int(item["task_type"])
    prediction = normalize_gold_answer(item.get("parse_response", ""), task_type)
    answer = normalize_gold_answer(item["answer"], task_type)

    if task_type == 3:
        return set(prediction.split(",")) == set(answer.split(","))
    return prediction.lower() == answer.lower()


class OpenAIParser:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_tokens=64,
        max_retries=3,
        retry_interval=2.0,
    ):
        if OpenAI is None:
            raise ImportError(
                "Install openai to enable LLM parsing: pip install openai"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def parse(self, item):
        task_type = int(item["task_type"])
        prompt = PARSE_PROMPTS[task_type].format(
            Response=item.get("model_response", "")
        )
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_interval)
        raise RuntimeError(f"LLM parser failed: {last_error}")


def should_skip_item(item):
    return bool(item.get("infer_error")) or not item.get("model_response")


def deduplicate(items):
    seen = set()
    deduped = []
    duplicate_count = 0

    for item in items:
        key = sample_key(item)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, duplicate_count


def parse_item(item, parser=None, llm_on_invalid=False, llm_on_all=False):
    parsed = dict(item)
    rule_answer = normalize_answer(parsed.get("model_response", ""), parsed)
    parsed["parse_response"] = rule_answer
    parsed["parse_method"] = "rule"

    if parser and (llm_on_all or (llm_on_invalid and rule_answer == "Invalid")):
        try:
            llm_answer = parser.parse(parsed)
            parsed["parse_response"] = normalize_answer(llm_answer, parsed)
            parsed["raw_parse_response"] = llm_answer
            parsed["parse_method"] = "llm"
        except Exception as exc:
            parsed["parse_error"] = str(exc)

    correct = is_correct(parsed)
    if correct is not None:
        parsed["correct"] = correct
    return parsed


def process_results(
    input_path,
    output_path,
    parser=None,
    num_workers=1,
    keep_failed=False,
    llm_on_invalid=False,
    llm_on_all=False,
):
    items = read_jsonl(input_path)
    skipped_failed = sum(1 for item in items if should_skip_item(item))
    if not keep_failed:
        items = [item for item in items if not should_skip_item(item)]

    items, duplicate_count = deduplicate(items)
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        parsed_items = [
            parse_item(
                item,
                parser=parser,
                llm_on_invalid=llm_on_invalid,
                llm_on_all=llm_on_all,
            )
            for item in tqdm(items, desc="Parsing")
        ]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    parse_item,
                    item,
                    parser=parser,
                    llm_on_invalid=llm_on_invalid,
                    llm_on_all=llm_on_all,
                )
                for item in items
            ]
            parsed_items = [future.result() for future in tqdm(futures, desc="Parsing")]

    write_jsonl(parsed_items, output_path)
    return build_summary(parsed_items, skipped_failed, duplicate_count)


def build_summary(items, skipped_failed=0, duplicate_count=0):
    summary = {
        "total": len(items),
        "skipped_failed": skipped_failed,
        "skipped_duplicates": duplicate_count,
        "invalid": 0,
        "with_answer": 0,
        "correct": 0,
        "accuracy": None,
        "task_stats": {},
    }

    for item in items:
        task_name = f"task_{int(item['task_type'])}"
        task_stats = summary["task_stats"].setdefault(
            task_name,
            {
                "total": 0,
                "invalid": 0,
                "with_answer": 0,
                "correct": 0,
                "accuracy": None,
            },
        )
        task_stats["total"] += 1

        if item.get("parse_response") == "Invalid":
            summary["invalid"] += 1
            task_stats["invalid"] += 1

        if "correct" in item:
            summary["with_answer"] += 1
            task_stats["with_answer"] += 1
            if item["correct"]:
                summary["correct"] += 1
                task_stats["correct"] += 1

    if summary["with_answer"]:
        summary["accuracy"] = summary["correct"] / summary["with_answer"]

    for task_stats in summary["task_stats"].values():
        if task_stats["with_answer"]:
            task_stats["accuracy"] = task_stats["correct"] / task_stats["with_answer"]

    return summary


def merge_summaries(summaries):
    merged = {
        "total": 0,
        "skipped_failed": 0,
        "skipped_duplicates": 0,
        "invalid": 0,
        "with_answer": 0,
        "correct": 0,
        "accuracy": None,
        "task_stats": {},
    }

    for summary in summaries:
        for key in (
            "total",
            "skipped_failed",
            "skipped_duplicates",
            "invalid",
            "with_answer",
            "correct",
        ):
            merged[key] += summary.get(key, 0)
        merged["task_stats"].update(summary.get("task_stats", {}))

    if merged["with_answer"]:
        merged["accuracy"] = merged["correct"] / merged["with_answer"]
    return merged


def build_parser(args):
    if not args.parser_model:
        return None

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if not api_key:
        raise ValueError(
            f"Set {args.api_key_env} or omit --parser-model to use rule parsing only."
        )

    return OpenAIParser(
        model=args.parser_model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
    )


def parse_task_ids(value):
    try:
        task_ids = [int(task_id) for task_id in value.split(",") if task_id.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--task-ids must be a comma-separated list of integers."
        ) from exc

    if not task_ids:
        raise argparse.ArgumentTypeError("--task-ids cannot be empty.")
    unsupported = [task_id for task_id in task_ids if task_id not in {1, 2, 3}]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported task id(s): {unsupported}. Expected values are 1, 2, 3."
        )
    return task_ids


def build_process_jobs(args):
    if bool(args.input_path) != bool(args.output_path):
        raise ValueError("--input-path and --output-path must be provided together.")

    if args.input_path and args.output_path:
        return [
            (None, Path(args.input_path), Path(args.output_path), args.summary_path)
        ]

    jobs = []
    for task_id in args.task_ids:
        input_path = Path(args.input_dir) / f"{args.input_prefix}_task{task_id}.jsonl"
        output_path = (
            Path(args.output_dir) / f"{args.output_prefix}_task{task_id}.parsed.jsonl"
        )
        summary_path = (
            Path(args.output_dir) / f"{args.output_prefix}_task{task_id}.summary.json"
        )
        jobs.append((task_id, input_path, output_path, summary_path))
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Process Artifact-Bench inference results generated by "
            "infer_qwen3_vl.py."
        )
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional inference result JSONL. If omitted, all selected tasks are run.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional processed result JSONL. Required when --input-path is used.",
    )
    parser.add_argument("--summary-path", default=None, help="Optional summary JSON.")
    parser.add_argument("--input-dir", default="results", help="Inference result dir.")
    parser.add_argument("--output-dir", default="results", help="Processed result dir.")
    parser.add_argument(
        "--task-ids",
        type=parse_task_ids,
        default=parse_task_ids("1,2,3"),
        help="Comma-separated task IDs to process in batch mode.",
    )
    parser.add_argument(
        "--input-prefix",
        default="qwen3_vl",
        help="Input filename prefix used in batch mode.",
    )
    parser.add_argument(
        "--output-prefix",
        default="qwen3_vl",
        help="Output filename prefix used in batch mode.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep failed samples.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--parser-model",
        default=None,
        help="Optional OpenAI-compatible model for answer extraction.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the parser API key.",
    )
    parser.add_argument("--base-url", default=None, help="Optional OpenAI base URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-interval", type=float, default=2.0)
    parser.add_argument(
        "--llm-on-invalid",
        action="store_true",
        help="Call the parser model only when rule parsing returns Invalid.",
    )
    parser.add_argument(
        "--llm-on-all",
        action="store_true",
        help="Call the parser model for every sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parser = build_parser(args)
    summaries = []
    jobs = build_process_jobs(args)

    for task_id, input_path, output_path, summary_path in jobs:
        print(f"Processing results: input={input_path}, output={output_path}")
        summary = process_results(
            input_path=input_path,
            output_path=output_path,
            parser=parser,
            num_workers=args.num_workers,
            keep_failed=args.keep_failed,
            llm_on_invalid=args.llm_on_invalid or bool(parser),
            llm_on_all=args.llm_on_all,
        )
        summaries.append(summary)

        if summary_path:
            write_json(summary, summary_path)

        accuracy = summary["accuracy"]
        accuracy_text = "N/A" if accuracy is None else f"{accuracy * 100:.2f}%"
        print(
            "Finished processing. "
            f"task={task_id or 'custom'}, "
            f"total={summary['total']}, "
            f"invalid={summary['invalid']}, "
            f"accuracy={accuracy_text}"
        )

    if len(summaries) == 1:
        return

    summary = merge_summaries(summaries)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.output_dir) / f"{args.output_prefix}_summary.json"
    )
    write_json(summary, summary_path)
    accuracy = summary["accuracy"]
    accuracy_text = "N/A" if accuracy is None else f"{accuracy * 100:.2f}%"
    print(
        "Finished all tasks. "
        f"total={summary['total']}, "
        f"invalid={summary['invalid']}, "
        f"accuracy={accuracy_text}, "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
