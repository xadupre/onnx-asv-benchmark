import argparse
import json
from pathlib import Path
import runpy


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
    cpu_backend_cases = [
        f"cpu_backend_cases/{category}/cases/{class_name}"
        for category, class_name, _, _, _ in case_shards
    ]
    return operators + modules + cpu_backend_cases


def main():
    parser = argparse.ArgumentParser(
        description="List all benchmark shards or those assigned to a schedule bucket."
    )
    parser.add_argument("bucket", nargs="?", type=int, choices=range(7))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
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
