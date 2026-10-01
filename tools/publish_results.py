import argparse
import subprocess
import tempfile
from pathlib import Path

from result_shards import migrate_legacy_results, write_shards


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True)


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
        help="Publish only this shard, for example models/tiny_llm; may be repeated.",
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
        destination = (checkout / args.subdirectory).resolve()
        if checkout.resolve() not in destination.parents:
            raise ValueError(
                "--subdirectory must remain inside the cache_data checkout"
            )
        destination.mkdir(parents=True, exist_ok=True)
        migrated = migrate_legacy_results(destination)
        published = write_shards(
            source,
            destination / "shards",
            selected_shards=args.shard or None,
        )

        run(["git", "add", "--all", "--", args.subdirectory], cwd=checkout)
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=checkout
        ).returncode
        if status == 0:
            print("No benchmark result changes to publish.")
        elif status == 1:
            labels = sorted(migrated | published)
            run(
                [
                    "git",
                    "commit",
                    "-m",
                    "Update onnx-asv-benchmark shards: " + ", ".join(labels),
                ],
                cwd=checkout,
            )
            for attempt in range(3):
                run(["git", "pull", "--rebase", "origin", "main"], cwd=checkout)
                push = subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=checkout,
                )
                if push.returncode == 0:
                    break
                if attempt == 2:
                    raise subprocess.CalledProcessError(
                        push.returncode,
                        ["git", "push", "origin", "main"],
                    )
        else:
            raise subprocess.CalledProcessError(
                status, ["git", "diff", "--cached", "--quiet"]
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
