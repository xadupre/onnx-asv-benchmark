import argparse
import json
from pathlib import Path


def discover_shards(root):
    benchmark_root = Path(root) / "benchmarks"
    categories = sorted(
        path.name
        for path in benchmark_root.iterdir()
        if path.is_dir()
        and path.name not in {"__pycache__", "models"}
        and not path.name.startswith(".")
    )
    models = sorted(
        f"models/{path.stem}"
        for path in (benchmark_root / "models").glob("*.py")
        if path.stem != "__init__"
    )
    return categories + models


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
