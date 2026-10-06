import argparse
import json
from pathlib import Path

from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS


def discover_shards(root):
    benchmark_root = Path(root) / "benchmarks"
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
        f"cpu_backend_cases/{category}/{module}/{class_name}"
        for category, module, class_name, _, _, _ in CASE_SHARDS
    ]
    return operators + modules + cpu_backend_cases


def main():
    parser = argparse.ArgumentParser(
        description="List the benchmark shards assigned to a weekly schedule bucket."
    )
    parser.add_argument("bucket", type=int, choices=range(7))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    shards = [
        shard
        for index, shard in enumerate(discover_shards(args.root))
        if index % 7 == args.bucket
    ]
    print(json.dumps({"shard": shards}, separators=(",", ":")))


if __name__ == "__main__":
    main()
