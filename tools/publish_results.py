import argparse
import subprocess
import tempfile
import time
from pathlib import Path

if __package__:
    from tools.result_shards import migrate_legacy_results, write_shards
else:
    from result_shards import migrate_legacy_results, write_shards


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True)


def publish_results(
    source, checkout, subdirectory, shards=None, machine=None, *, retry_delay=5
):
    destination = (checkout / subdirectory).resolve()
    if checkout.resolve() not in destination.parents:
        raise ValueError("--subdirectory must remain inside the cache_data checkout")

    max_attempts = 5
    for attempt in range(max_attempts):
        run(["git", "fetch", "origin", "main"], cwd=checkout)
        run(["git", "reset", "--hard", "origin/main"], cwd=checkout)
        destination.mkdir(parents=True, exist_ok=True)
        history = destination / "publication-history"
        try:
            migrated = migrate_legacy_results(destination, history_root=history)
            published = write_shards(
                source,
                destination / "shards",
                selected_shards=shards,
                machine=machine,
                history_root=history,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot reconcile benchmark results: {exc} "
                f"Local results remain in {source}; inspect the destination metadata "
                "before retrying publication."
            ) from exc

        run(["git", "add", "--all", "--", subdirectory], cwd=checkout)
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=checkout
        ).returncode
        if status == 0:
            print("No benchmark result changes to publish.")
            return
        if status != 1:
            raise subprocess.CalledProcessError(
                status, ["git", "diff", "--cached", "--quiet"]
            )
        run(
            [
                "git",
                "commit",
                "-m",
                "Update onnx-asv-benchmark shards: "
                + ", ".join(sorted(migrated | published)),
            ],
            cwd=checkout,
        )
        push = subprocess.run(["git", "push", "origin", "main"], cwd=checkout)
        if push.returncode == 0:
            return
        if attempt + 1 < max_attempts:
            delay = retry_delay * 2**attempt
            print(
                f"Publication push failed (attempt {attempt + 1}/{max_attempts}); "
                f"retrying in {delay}s before refreshing results.",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Benchmark publication failed after {max_attempts} push attempts. "
        f"Local results remain in {source}; check repository write access and "
        "concurrent publishers, then rerun tools/publish_results.py."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Publish local ASV results to the shared cache_data repository."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(".asv/results"),
        help="ASV results directory (default: .asv/results).",
    )
    parser.add_argument(
        "--repository",
        default="https://github.com/xadupre/cache_data.git",
        help="Git repository receiving the results.",
    )
    parser.add_argument(
        "--subdirectory",
        default="onnx-asv-benchmark",
        help="Destination subdirectory in the repository.",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help=(
            "Publish only this shard, for example models/llm/tiny_llm; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--machine",
        help="Publish results only for this ASV machine profile.",
    )
    parser.add_argument(
        "--pages-repository",
        default="xadupre/onnx-asv-benchmark",
        help="GitHub repository hosting the ASV website.",
    )
    parser.add_argument(
        "--publish-pages",
        action="store_true",
        help="Trigger the GitHub Pages deployment workflow after publishing.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"ASV results directory does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="onnx-asv-publish-") as temporary:
        checkout = Path(temporary) / "cache_data"
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                args.repository,
                str(checkout),
            ],
            cwd=Path.cwd(),
        )
        publish_results(
            source,
            checkout,
            args.subdirectory,
            shards=args.shard or None,
            machine=args.machine,
        )

    if args.publish_pages:
        run(
            [
                "gh",
                "workflow",
                "run",
                "publish-pages.yml",
                "--repo",
                args.pages_repository,
                "--ref",
                "main",
            ],
            cwd=Path.cwd(),
        )


if __name__ == "__main__":
    main()
