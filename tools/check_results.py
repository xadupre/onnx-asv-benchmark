import argparse
import ast
import itertools
import json
import math
from pathlib import Path

from tools.result_shards import benchmark_shard


def _has_successful_value(value):
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_successful_value(item) for item in value)
    return True


def successful_benchmark_count(source, shard):
    source = Path(source)
    successful = 0
    for result_path in source.glob("*/*.json"):
        if result_path.name == "machine.json":
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        columns = result.get("result_columns", ["result"])
        result_index = columns.index("result")
        for name, values in result.get("results", {}).items():
            if benchmark_shard(name) != shard:
                continue
            value = values[result_index] if len(columns) > 1 else values
            if _has_successful_value(value):
                successful += 1
    return successful


def failed_qwen_bf16_cases(source):
    failures = []
    checked = set()
    for result_path in Path(source).glob("*/*.json"):
        if result_path.name == "machine.json":
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        columns = result.get("result_columns", ["result"])
        result_index = columns.index("result")
        for name, values in result.get("results", {}).items():
            if not name.endswith((".Qwen2.time_prefill", ".Qwen2.time_decode")):
                continue
            timings = values[result_index] if len(columns) > 1 else values
            params = values[columns.index("params")] if "params" in columns else None
            if params is None:
                raise ValueError(f"Missing ASV parameters for {name!r}.")
            dimensions = [[ast.literal_eval(item) for item in axis] for axis in params]
            cases = list(itertools.product(*dimensions))
            if len(cases) != len(timings):
                raise ValueError(f"Unexpected ASV result count for {name!r}.")
            for case, timing in zip(cases, timings, strict=True):
                if case[-2] == "bf16" and case[-1] in {"onnx-light", "onnx-light-cpu"}:
                    checked.add((name.rsplit(".", 1)[-1], case[-1]))
                    if not _has_successful_value(timing):
                        failures.append(f"{name}: {case[-2]} / {case[-1]}")
    for method, backend in itertools.product(
        ("time_prefill", "time_decode"), ("onnx-light", "onnx-light-cpu")
    ):
        if (method, backend) not in checked:
            failures.append(f"Qwen2.{method}: missing bf16 / {backend} result")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Require at least one successful benchmark in an ASV shard."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("shard")
    args = parser.parse_args()

    if args.shard == "models/llm/qwen2":
        failures = failed_qwen_bf16_cases(args.source)
        if failures:
            raise RuntimeError(
                "Qwen2 BF16 setup or timing failed (ONNX Runtime BF16 CPU is "
                "intentionally unavailable): " + ", ".join(failures)
            )
    count = successful_benchmark_count(args.source, args.shard)
    if count == 0:
        raise RuntimeError(f"Every benchmark failed in shard {args.shard!r}.")
    print(f"{count} benchmark(s) produced at least one result in {args.shard!r}.")


if __name__ == "__main__":
    main()
