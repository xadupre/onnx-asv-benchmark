"""Publish ASV benchmark errors alongside the generated graph JSON."""

import argparse
from itertools import product
import json
from pathlib import Path

from asv.graph import Graph
from asv.util import sanitize_filename


def publish_errors(results_dir, html_dir):
    results_dir, html_dir = Path(results_dir), Path(html_dir)
    index_path = html_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    revisions = {commit: revision for revision, commit in index["revision_to_hash"].items()}
    graphs = index["graph_param_list"]
    errors = {}

    for result_path in results_dir.glob("*/*.json"):
        if result_path.name == "machine.json":
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        revision = revisions.get(result["commit_hash"])
        if revision is None:
            continue
        parameters = {
            **result["params"],
            **{f"env-{key}": value for key, value in result.get("env_vars", {}).items()},
        }
        for name, failures in result.get("benchmark_errors", {}).items():
            if name not in index["benchmarks"]:
                continue
            columns = result["result_columns"]
            old_params = result["results"][name][columns.index("params")]
            new_params = index["benchmarks"][name]["params"]
            old_combinations = list(product(*old_params))
            new_indices = {
                combination: position
                for position, combination in enumerate(product(*new_params))
            }
            aligned = {
                str(new_indices[old_combinations[int(position)]]): message
                for position, message in failures.items()
                if int(position) < len(old_combinations)
                and old_combinations[int(position)] in new_indices
            }
            if not aligned:
                continue
            for graph in graphs:
                if any(
                    parameters.get(key) != value
                    for key, value in graph.items()
                    if key != "branch"
                ):
                    continue
                path = Graph.get_file_path(graph, name) + ".json"
                if not (html_dir / path).is_file():
                    continue
                errors.setdefault(name, {}).setdefault(path, {}).setdefault(
                    revision, {}
                ).update(aligned)

    error_dir = html_dir / "errors"
    error_dir.mkdir(exist_ok=True)
    for name, failures in errors.items():
        (error_dir / f"{sanitize_filename(name)}.json").write_text(
            json.dumps(failures, sort_keys=True), encoding="utf-8"
        )
    index["error_benchmarks"] = sorted(errors)
    index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("html_dir", type=Path)
    args = parser.parse_args()
    publish_errors(args.results_dir, args.html_dir)


if __name__ == "__main__":
    main()
