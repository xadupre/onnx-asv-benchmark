import argparse
import json
from pathlib import Path


def remove_track_benchmarks(path):
    path = Path(path)
    benchmarks = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = {
        name: benchmark
        for name, benchmark in benchmarks.items()
        if not (
            isinstance(benchmark, dict)
            and benchmark.get("type") == "track"
            and name.startswith(("machine.track_", "versions.track_"))
        )
    }
    path.write_text(json.dumps(benchmarks), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove legacy machine and version tracks before publishing."
    )
    parser.add_argument("benchmarks", type=Path)
    remove_track_benchmarks(parser.parse_args().benchmarks)
