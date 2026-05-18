import argparse
import json
import multiprocessing as mp
import os
import queue
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .prompt import PROMPT_TASK_1, PROMPT_TASK_2, build_task3_prompt
except ImportError:
    from prompt import PROMPT_TASK_1, PROMPT_TASK_2, build_task3_prompt


VIDEO_PAIR_KEYS = (
    ("video_path_a", "video_path_b"),
    ("video_path_A", "video_path_B"),
    ("video_a_path", "video_b_path"),
    ("video_A_path", "video_B_path"),
    ("video_a", "video_b"),
    ("video_A", "video_B"),
    ("Video A", "Video B"),
)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(item, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def get_video_paths(item):
    if "video_path" in item:
        video_path = item["video_path"]
        if isinstance(video_path, (list, tuple)):
            return list(video_path)
        return [video_path]

    for key_a, key_b in VIDEO_PAIR_KEYS:
        if key_a in item and key_b in item:
            return [item[key_a], item[key_b]]

    raise KeyError("Cannot find video path field(s) in sample.")


def sample_key(item):
    video_paths = get_video_paths(item)
    return video_paths[0] if len(video_paths) == 1 else json.dumps(video_paths)


def build_prompt(item):
    task_type = int(item["task_type"])
    if task_type == 1:
        return PROMPT_TASK_1
    if task_type == 2:
        return PROMPT_TASK_2
    if task_type == 3:
        return build_task3_prompt(item.get("option"))
    raise ValueError(f"Unsupported task_type: {task_type}")


def build_messages(video_paths, prompt, task_type):
    if task_type in (1, 3):
        if len(video_paths) != 1:
            raise ValueError(
                f"Task {task_type} expects one video, got {len(video_paths)}."
            )
        content = [
            {"type": "video", "video": video_paths[0]},
            {"type": "text", "text": prompt},
        ]
    elif task_type == 2:
        if len(video_paths) != 2:
            raise ValueError(f"Task 2 expects two videos, got {len(video_paths)}.")
        content = [
            {"type": "text", "text": "Video A: "},
            {"type": "video", "video": video_paths[0]},
            {"type": "text", "text": "\nVideo B: "},
            {"type": "video", "video": video_paths[1]},
            {"type": "text", "text": f"\n{prompt}"},
        ]
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return [{"role": "user", "content": content}]


def load_completed_results(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return {}

    completed = {}
    with output_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                completed[sample_key(item)] = item
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"Warning: skip invalid line {line_no} in {output_path}: {exc}")
    return completed


class Qwen3VLInfer:
    def __init__(self, model_path, gpu_id, attn_implementation="flash_attention_2"):
        self.device = f"cuda:{gpu_id}"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map={"": self.device},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.inference_mode()
    def infer(self, sample, fps, max_new_tokens):
        messages = build_messages(
            video_paths=get_video_paths(sample),
            prompt=build_prompt(sample),
            task_type=int(sample["task_type"]),
        )
        inputs = self.processor.apply_chat_template(
            messages,
            fps=fps,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def infer_worker(
    worker_id,
    gpu_id,
    model_path,
    tasks,
    result_queue,
    status_queue,
    fps,
    max_new_tokens,
    attn_implementation,
):
    try:
        torch.cuda.set_device(gpu_id)
        engine = Qwen3VLInfer(model_path, gpu_id, attn_implementation)
        status_queue.put(("ready", worker_id, gpu_id))
    except Exception as exc:
        status_queue.put(("load_error", worker_id, gpu_id, str(exc)))
        return

    for index, sample in tasks:
        try:
            result_queue.put(
                (
                    index,
                    engine.infer(sample, fps, max_new_tokens),
                    None,
                    worker_id,
                    gpu_id,
                )
            )
        except Exception as exc:
            result_queue.put((index, "", str(exc), worker_id, gpu_id))


def split_tasks(tasks, num_workers):
    splits = [[] for _ in range(num_workers)]
    for task_index, task in enumerate(tasks):
        splits[task_index % num_workers].append(task)
    return splits


def wait_until_ready(workers, status_queue, num_workers):
    ready = 0
    progress = tqdm(total=num_workers, desc="Loading models", dynamic_ncols=True)
    try:
        while ready < num_workers:
            try:
                status = status_queue.get(timeout=5)
            except queue.Empty:
                dead_workers = [
                    worker.pid
                    for worker in workers
                    if not worker.is_alive() and worker.exitcode not in (0, None)
                ]
                if dead_workers:
                    raise RuntimeError(
                        f"Worker crashed during model loading: {dead_workers}"
                    )
                continue

            if status[0] == "ready":
                _, worker_id, gpu_id = status
                ready += 1
                progress.update(1)
                progress.set_postfix_str(f"worker={worker_id}, gpu={gpu_id}")
            elif status[0] == "load_error":
                _, worker_id, gpu_id, error = status
                raise RuntimeError(
                    f"Worker {worker_id} on GPU {gpu_id} failed to load model: {error}"
                )
    finally:
        progress.close()


def run_parallel_inference(
    model_path,
    input_path,
    output_path,
    gpu_ids,
    fps=5,
    max_new_tokens=8192,
    attn_implementation="flash_attention_2",
    eval_info=None,
):
    samples = read_jsonl(input_path)
    completed = load_completed_results(output_path)
    results = [None] * len(samples)
    pending = []

    for index, sample in enumerate(samples):
        key = sample_key(sample)
        if key in completed:
            results[index] = completed[key]
        else:
            pending.append((index, sample))

    if not pending:
        return results, len(completed), 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    visible_gpus = torch.cuda.device_count()
    if max(gpu_ids) >= visible_gpus:
        raise RuntimeError(
            f"Requested GPU IDs {gpu_ids}, but only {visible_gpus} GPU(s) are visible."
        )

    gpu_ids = gpu_ids[: min(len(gpu_ids), len(pending))]
    worker_tasks = split_tasks(pending, len(gpu_ids))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    status_queue = ctx.Queue()
    workers = []

    for worker_id, gpu_id in enumerate(gpu_ids):
        worker = ctx.Process(
            target=infer_worker,
            args=(
                worker_id,
                gpu_id,
                model_path,
                worker_tasks[worker_id],
                result_queue,
                status_queue,
                fps,
                max_new_tokens,
                attn_implementation,
            ),
        )
        worker.start()
        workers.append(worker)

    wait_until_ready(workers, status_queue, len(workers))

    failed = 0
    worker_done = [0] * len(workers)
    worker_bars = [
        tqdm(
            total=len(worker_tasks[worker_id]),
            desc=f"GPU {gpu_id}",
            position=worker_id,
            leave=True,
            dynamic_ncols=True,
        )
        for worker_id, gpu_id in enumerate(gpu_ids)
    ]
    total_bar = tqdm(
        total=len(pending),
        desc="Inferencing",
        position=len(workers),
        leave=True,
        dynamic_ncols=True,
    )

    try:
        finished = 0
        while finished < len(pending):
            try:
                index, answer, error, worker_id, gpu_id = result_queue.get(timeout=5)
            except queue.Empty:
                dead_workers = [
                    worker.pid
                    for worker in workers
                    if not worker.is_alive() and worker.exitcode not in (0, None)
                ]
                if dead_workers:
                    raise RuntimeError(
                        f"Worker crashed before finishing all tasks: {dead_workers}"
                    )
                continue

            result = dict(samples[index])
            result["model_response"] = answer
            if error:
                result["infer_error"] = error
                failed += 1
            if eval_info:
                result["eval_info"] = dict(eval_info)

            results[index] = result
            append_jsonl(result, output_path)

            finished += 1
            worker_done[worker_id] += 1
            worker_bars[worker_id].update(1)
            worker_bars[worker_id].set_postfix(
                done=worker_done[worker_id],
                failed=failed,
            )
            total_bar.update(1)
            total_bar.set_postfix(failed=failed, skipped=len(completed), gpu=gpu_id)
    finally:
        total_bar.close()
        for bar in worker_bars:
            bar.close()

    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(
                f"Worker process {worker.pid} exited with code {worker.exitcode}."
            )

    return results, len(completed), len(pending)


def parse_gpu_ids(value):
    try:
        gpu_ids = [int(gpu_id) for gpu_id in value.split(",") if gpu_id.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be a comma-separated list of integers."
        ) from exc

    if not gpu_ids:
        raise argparse.ArgumentTypeError("--gpu-ids cannot be empty.")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids cannot contain negative values.")
    return gpu_ids


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


def build_eval_jobs(args):
    if bool(args.input_path) != bool(args.output_path):
        raise ValueError("--input-path and --output-path must be provided together.")

    if args.input_path and args.output_path:
        return [(None, Path(args.input_path), Path(args.output_path))]

    jobs = []
    for task_id in args.task_ids:
        input_path = Path(args.task_dir) / f"task{task_id}_meta.jsonl"
        output_path = (
            Path(args.output_dir) / f"{args.output_prefix}_task{task_id}.jsonl"
        )
        jobs.append((task_id, input_path, output_path))
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference on Artifact-Bench JSONL files."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path or HF model ID.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional input JSONL file. If omitted, all selected tasks are run.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output JSONL file. Required when --input-path is used.",
    )
    parser.add_argument("--task-dir", default="task", help="Task metadata directory.")
    parser.add_argument("--output-dir", default="results", help="Output directory.")
    parser.add_argument(
        "--task-ids",
        type=parse_task_ids,
        default=parse_task_ids("1,2,3"),
        help="Comma-separated task IDs to run in batch mode.",
    )
    parser.add_argument(
        "--output-prefix",
        default="qwen3_vl",
        help="Output filename prefix used in batch mode.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=parse_gpu_ids("0"),
        help="Comma-separated GPU IDs.",
    )
    parser.add_argument("--fps", type=int, default=5, help="Video sampling FPS.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional name saved in eval_info.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation passed to transformers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_eval_info = {
        "model": args.model_name or Path(args.model_path).name,
        "fps": args.fps,
        "max_new_tokens": args.max_new_tokens,
    }
    for task_id, input_path, output_path in build_eval_jobs(args):
        eval_info = dict(base_eval_info)
        if task_id is not None:
            eval_info["task_id"] = task_id

        print(f"Running inference: input={input_path}, output={output_path}")
        _, skipped, processed = run_parallel_inference(
            model_path=args.model_path,
            input_path=input_path,
            output_path=output_path,
            gpu_ids=args.gpu_ids,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
            eval_info=eval_info,
        )
        print(
            "Finish evaluation. "
            f"task={task_id or 'custom'}, "
            f"skipped={skipped}, "
            f"newly_processed={processed}"
        )


if __name__ == "__main__":
    main()
