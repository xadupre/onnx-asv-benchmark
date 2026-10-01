import argparse
import json
from pathlib import Path


def remove_track_benchmarks(path):
    path = Path(path)
    benchmarks = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = {
        name: benchmark
        for name, benchmark in benchmarks.items()
        if not isinstance(benchmark, dict) or benchmark.get("type") != "track"
    }
    path.write_text(json.dumps(benchmarks), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove ASV track benchmarks before publishing the site."
    )
    parser.add_argument("benchmarks", type=Path)
    remove_track_benchmarks(parser.parse_args().benchmarks)
