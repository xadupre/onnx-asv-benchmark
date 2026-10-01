import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Require at least one successful benchmark in an ASV shard."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("shard")
    args = parser.parse_args()

    count = successful_benchmark_count(args.source, args.shard)
    if count == 0:
        raise RuntimeError(f"Every benchmark failed in shard {args.shard!r}.")
    print(f"{count} benchmark(s) produced at least one result in {args.shard!r}.")


if __name__ == "__main__":
    main()
