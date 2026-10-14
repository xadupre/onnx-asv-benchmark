import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import runpy


@lru_cache(maxsize=1)
def case_shard_classes():
    benchmark_root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = runpy.run_path(benchmark_root / "cpu_backend_cases" / "_manifest.py")
    grouped = {}
    for category, class_name, prefix, _, _ in manifest["CASE_SHARDS"]:
        grouped.setdefault(f"cpu_backend_cases/{category}/cases/{prefix}", []).append(
            class_name
        )
    return grouped


@lru_cache(maxsize=1)
def case_class_shards():
    return {
        (shard.split("/")[1], class_name): shard
        for shard, classes in case_shard_classes().items()
        for class_name in classes
    }


def shard_benchmark_regex(shard):
    if shard in case_shard_classes():
        category = shard.split("/")[1]
        classes = case_shard_classes()[shard]
        return (
            "^"
            + re.escape(f"cpu_backend_cases.{category}.cases.")
            + "(?:"
            + "|".join(re.escape(name) for name in classes)
            + r")\."
        )
    return "^" + re.escape(shard.replace("/", ".")) + r"\."


def discover_shards(root):
    benchmark_root = Path(root) / "benchmarks"
    case_shards = runpy.run_path(benchmark_root / "cpu_backend_cases" / "_manifest.py")[
        "CASE_SHARDS"
    ]
    operators = sorted(
        path.relative_to(benchmark_root).with_suffix("").as_posix()
        for path in (benchmark_root / "ops").glob("*/*.py")
        if not path.name.startswith("_")
    )
    modules = sorted(
        path.relative_to(benchmark_root).with_suffix("").as_posix()
        for root_name in ("builder", "models")
        for path in (benchmark_root / root_name).rglob("*.py")
        if not path.name.startswith("_")
    )
    cpu_backend_cases = sorted({
        f"cpu_backend_cases/{category}/cases/{prefix}"
        for category, _, prefix, _, _ in case_shards
    })
    return operators + modules + cpu_backend_cases


def main():
    parser = argparse.ArgumentParser(
        description="List all benchmark shards or those assigned to a schedule bucket."
    )
    parser.add_argument("bucket", nargs="?", type=int, choices=range(7))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--regex", help="Print the ASV filter for one shard.")
    args = parser.parse_args()
    if args.regex:
        print(shard_benchmark_regex(args.regex))
        return
    discovered = discover_shards(args.root)
    if args.bucket is None:
        print("\n".join(discovered))
        return
    shards = [
        shard
        for index, shard in enumerate(discovered)
        if index % 7 == args.bucket
    ]
    print(json.dumps({"shard": shards}, separators=(",", ":")))


if __name__ == "__main__":
    main()
