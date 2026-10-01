import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASV benchmarks in the current Python environment.",
        epilog="""examples:
  python tools/run_asv.py
  python tools/run_asv.py --quick
  python tools/run_asv.py --bench MatMul main^!""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "range",
        nargs="?",
        help="Git commit range to benchmark (default: configured branches).",
    )
    parser.add_argument(
        "-b",
        "--bench",
        action="append",
        default=[],
        metavar="REGEX",
        help="Run benchmarks whose names match REGEX; may be repeated.",
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Run each benchmark once without saving results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processor = subprocess.check_output(
        [sys.executable, str(root / "tools" / "setup_machine.py")],
        text=True,
    ).strip()
    env = {**os.environ, "ASV_PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    command = [
        sys.executable,
        "-m",
        "asv",
        "run",
        "--environment",
        f"existing:{sys.executable}",
        "--machine",
        processor,
    ]
    for benchmark in args.bench:
        command.extend(["--bench", benchmark])
    if args.quick:
        command.append("--quick")
    if args.range:
        command.append(args.range)
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
