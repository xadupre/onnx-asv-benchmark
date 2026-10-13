import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def shard_benchmark_regex(shard):
    return "^" + re.escape(shard.replace("/", ".")) + r"\."


def resolve_commit(root, commit):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=root,
        text=True,
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASV benchmarks in the current Python environment.",
        epilog="""examples:
  python tools/run_asv.py
  python tools/run_asv.py --quick
  python tools/run_asv.py --bench MatMul
  python tools/run_asv.py --publish --shard models/llm/tiny_llm""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "commit",
        nargs="?",
        default="HEAD",
        help="Git commit to associate with the results (default: HEAD).",
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
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish saved results to cache_data after a successful run.",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        metavar="NAME",
        help="Run only this shard and publish it with --publish; may be repeated.",
    )
    args = parser.parse_args()
    if args.publish and args.quick:
        parser.error(
            "--publish cannot be combined with --quick because quick runs are not saved"
        )
    if args.bench and args.shard:
        parser.error(
            "--bench cannot be combined with --shard; --shard selects the "
            "matching benchmarks automatically"
        )
    if args.bench and args.publish:
        parser.error(
            "--bench cannot be combined with --publish; use --shard to select "
            "and publish matching benchmarks"
        )
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    commit_hash = resolve_commit(root, args.commit)
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
        "--set-commit-hash",
        commit_hash,
    ]
    for benchmark in args.bench:
        command.extend(["--bench", benchmark])
    for shard in args.shard:
        command.extend(["--bench", shard_benchmark_regex(shard)])
    if args.quick:
        command.append("--quick")
    status = subprocess.call(command, cwd=root, env=env)
    if status not in (0, 2):
        raise SystemExit(status)
    if args.publish:
        publish_command = [
            sys.executable,
            str(root / "tools" / "publish_results.py"),
        ]
        for shard in args.shard:
            publish_command.extend(["--shard", shard])
        subprocess.run(publish_command, cwd=root, env=env, check=True)
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
