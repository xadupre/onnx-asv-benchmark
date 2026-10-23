import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import runpy


BATCH_SIZES = {
    "cpu_backend_cases/math/cases/add": 6,
    "cpu_backend_cases/math/cases/mul": 6,
    "cpu_backend_cases/logical/cases/greaterorequal": 6,
    "cpu_backend_cases/math/cases/mod": 10,
    "cpu_backend_cases/math/cases/pow": 10,
    "cpu_backend_cases/logical/cases/greater": 8,
    "cpu_backend_cases/logical/cases/less": 10,
    "cpu_backend_cases/math/cases/div": 6,
    "cpu_backend_cases/logical/cases/bitwiseor": 10,
    "cpu_backend_cases/logical/cases/bitwisexor": 10,
    "cpu_backend_cases/math/cases/prelu": 10,
    "cpu_backend_cases/logical/cases/bitwiseand": 10,
    "cpu_backend_cases/logical/cases/lessorequal": 10,
    "cpu_backend_cases/math/cases/sub": 6,
    "cpu_backend_cases/logical/cases/equal": 10,
}


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


def batch_count(shard):
    size = BATCH_SIZES.get(shard)
    return (len(case_shard_classes()[shard]) + size - 1) // size if size else 1


def batch_matrix(shards):
    return {
        "include": [
            {"shard": shard, "batch": batch}
            for shard in shards
            for batch in range(batch_count(shard))
        ]
    }


def shard_benchmark_regex(shard, batch=None):
    if batch is not None and not 0 <= batch < batch_count(shard):
        raise ValueError(f"Invalid batch {batch} for shard {shard!r}.")
    if shard in case_shard_classes():
        category = shard.split("/")[1]
        classes = case_shard_classes()[shard]
        if batch is not None and shard in BATCH_SIZES:
            size = BATCH_SIZES[shard]
            classes = classes[batch * size : (batch + 1) * size]
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
    parser.add_argument("--batch", type=int, help="Batch index for --regex.")
    parser.add_argument("--matrix", action="store_true", help="Print a batched job matrix.")
    parser.add_argument("--requested", help="Comma-separated shards for --matrix.")
    args = parser.parse_args()
    if args.regex:
        print(shard_benchmark_regex(args.regex, args.batch))
        return
    if args.requested and not args.matrix:
        parser.error("--requested requires --matrix")
    if args.batch is not None:
        parser.error("--batch requires --regex")
    discovered = (
        [shard.strip() for shard in args.requested.split(",") if shard.strip()]
        if args.requested is not None
        else discover_shards(args.root)
    )
    if not discovered:
        parser.error("No shards requested")
    if args.bucket is None:
        if args.matrix:
            if args.requested is None:
                parser.error("--matrix requires a bucket or --requested")
            print(json.dumps(batch_matrix(discovered), separators=(",", ":")))
            return
        print("\n".join(discovered))
        return
    shards = [
        shard
        for index, shard in enumerate(discovered)
        if index % 7 == args.bucket
    ]
    if args.matrix:
        print(json.dumps(batch_matrix(shards), separators=(",", ":")))
    else:
        print(json.dumps({"shard": shards}, separators=(",", ":")))


if __name__ == "__main__":
    main()
