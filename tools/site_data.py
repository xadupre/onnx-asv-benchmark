import argparse
import itertools
import json
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r} in ASV output.")
    return text.replace(old, new)


def write_configuration_previews(html_dir):
    from asv.graph import Graph

    html_dir = Path(html_dir)
    index = json.loads((html_dir / "index.json").read_text(encoding="utf-8"))
    graph_parameters = {
        Path(Graph.get_file_path(params, "_")).parent: params
        for params in index["graph_param_list"]
    }
    benchmark_graphs = {}
    for path in (html_dir / "graphs").rglob("*.json"):
        parameters = graph_parameters.get(path.relative_to(html_dir).parent)
        if parameters is not None:
            benchmark_graphs.setdefault(path.stem, []).append((path, parameters))

    for name, benchmark in index["benchmarks"].items():
        combinations = list(itertools.product(*benchmark["params"]))
        series = []
        configurations = []
        for path, parameters in sorted(benchmark_graphs.get(name, [])):
            graph_index = len(configurations)
            configurations.append(parameters)
            points = json.loads(path.read_text(encoding="utf-8"))
            curves = [[] for _ in combinations]
            for revision, values in points:
                if not isinstance(values, list):
                    values = (
                        [values] * len(combinations) if values is None else [values]
                    )
                if len(values) != len(combinations):
                    raise ValueError(
                        f"Incompatible benchmark parameter count in {path}."
                    )
                date = index["revision_to_date"][str(revision)]
                for curve, value in zip(curves, values):
                    curve.append([date, value])
            for combination, curve in zip(combinations, curves):
                series.append(
                    {
                        "graph": graph_index,
                        "parameters": dict(zip(benchmark["param_names"], combination)),
                        "data": sorted(curve),
                    }
                )
        summary = html_dir / "graphs" / "configurations" / f"{name}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {"graph_parameters": configurations, "series": series},
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return len(index["benchmarks"])


def extract_site_data(html_dir, destination):
    html_dir = Path(html_dir)
    destination = Path(destination)
    json_files = sorted(html_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No ASV JSON data found in {html_dir}.")
    if destination.is_dir():
        shutil.rmtree(destination)
    for source in json_files:
        target = destination / source.relative_to(html_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(json_files) + write_configuration_previews(destination)


def configure_remote_data(html_dir, base_url):
    html_dir = Path(html_dir)
    base_url = base_url.rstrip("/") + "/"
    if not base_url.startswith("https://"):
        raise ValueError("The ASV data URL must use HTTPS.")

    script_path = html_dir / "asv.js"
    script = script_path.read_text(encoding="utf-8")
    script = replace_once(
        script,
        "'use strict';",
        "'use strict';\n\n"
        f"var asv_data_base_url = {json.dumps(base_url)};\n"
        "function asv_data_path(path) {\n"
        "    return new URL(path, asv_data_base_url).toString();\n"
        "}",
    )
    script = replace_once(
        script,
        "url: url + '?timestamp='",
        "url: asv_data_path(url) + '?timestamp='",
    )
    script = replace_once(
        script,
        "url: \"index.json\" + '?timestamp='",
        "url: asv_data_path(\"index.json\") + '?timestamp='",
    )
    script = replace_once(
        script,
        'url: "info.json",',
        'url: asv_data_path("info.json"),',
    )
    script_path.write_text(script, encoding="utf-8")

    regressions_path = html_dir / "regressions.js"
    regressions = regressions_path.read_text(encoding="utf-8")
    regressions = replace_once(
        regressions,
        "url: 'regressions.json' + '?timestamp='",
        "url: asv_data_path('regressions.json') + '?timestamp='",
    )
    regressions_path.write_text(regressions, encoding="utf-8")

    for path in html_dir.rglob("*.json"):
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Extract ASV JSON data or configure its HTML to load remote data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("html_dir", type=Path)
    extract.add_argument("destination", type=Path)

    configure = subparsers.add_parser("configure")
    configure.add_argument("html_dir", type=Path)
    configure.add_argument("base_url")

    args = parser.parse_args()
    if args.command == "extract":
        count = extract_site_data(args.html_dir, args.destination)
        print(f"Copied {count} ASV JSON files to {args.destination}.")
    else:
        configure_remote_data(args.html_dir, args.base_url)


if __name__ == "__main__":
    main()
