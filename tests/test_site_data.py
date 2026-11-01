import json
import tempfile
import unittest
from pathlib import Path

from tools.site_data import (
    configure_remote_data,
    extract_site_data,
    write_configuration_previews,
)

ASV_SCRIPT = """'use strict';
function load_graph_data(url) {
    $.ajax({
        url: url + '?timestamp=' + $.asv.main_timestamp,
        dataType: "json"
    });
}
function init_index() {
    $.ajax({
        url: "index.json" + '?timestamp=' + $.asv.main_timestamp,
        dataType: "json"
    });
}
function init() {
    $.ajax({
        url: "info.json",
        dataType: "json"
    });
}
"""


class TestSiteData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.html = self.root / "html"
        self.html.mkdir()
        (self.html / "asv.js").write_text(ASV_SCRIPT, encoding="utf-8")
        (self.html / "regressions.js").write_text(
            "$.ajax({url: 'regressions.json' + '?timestamp=' + timestamp});",
            encoding="utf-8",
        )
        (self.html / "index.html").write_text("<html></html>", encoding="utf-8")
        (self.html / "index.json").write_text(
            json.dumps(
                {"project": "benchmark", "benchmarks": {}, "graph_param_list": []}
            ),
            encoding="utf-8",
        )
        graph = self.html / "graphs" / "math.json"
        graph.parent.mkdir()
        graph.write_text(json.dumps([[1, 2]]), encoding="utf-8")

    def test_extract_and_configure(self):
        destination = self.root / "site-data"
        stale = destination / "stale.json"
        stale.parent.mkdir()
        stale.write_text("{}", encoding="utf-8")

        self.assertEqual(extract_site_data(self.html, destination), 2)
        self.assertFalse(stale.exists())
        self.assertTrue((destination / "index.json").is_file())
        self.assertTrue((destination / "graphs" / "math.json").is_file())

        configure_remote_data(
            self.html,
            "https://raw.githubusercontent.com/example/cache/main/site-data",
        )

        script = (self.html / "asv.js").read_text(encoding="utf-8")
        self.assertIn(
            '"https://raw.githubusercontent.com/example/cache/main/site-data/"',
            script,
        )
        self.assertIn("url: asv_data_path(url) + '?timestamp='", script)
        self.assertIn('url: asv_data_path("index.json")', script)
        self.assertIn('url: asv_data_path("info.json")', script)
        self.assertIn(
            "url: asv_data_path('regressions.json')",
            (self.html / "regressions.js").read_text(encoding="utf-8"),
        )
        self.assertFalse(list(self.html.rglob("*.json")))
        self.assertTrue((self.html / "index.html").is_file())

    def test_requires_asv_json(self):
        for path in self.html.rglob("*.json"):
            path.unlink()
        with self.assertRaises(FileNotFoundError):
            extract_site_data(self.html, self.root / "site-data")

    def test_requires_https(self):
        with self.assertRaises(ValueError):
            configure_remote_data(self.html, "http://example.com/site-data")

    def test_previews_separate_machines_and_every_parameter(self):
        from asv.graph import Graph

        name = "models.test.Model.time_run"
        machines = [
            {"machine": "slow", "python": "3.12"},
            {"machine": "fast", "python": "3.12"},
            {"machine": "slow", "python": "3.13"},
        ]
        index = {
            "benchmarks": {
                name: {
                    "param_names": ["dtype", "backend"],
                    "params": [["'float32'", "'float64'"], ["'ort'", "'reference'"]],
                }
            },
            "graph_param_list": machines,
            "revision_to_date": {"1": 1000, "2": 2000, "3": 3000},
        }
        (self.html / "index.json").write_text(json.dumps(index), encoding="utf-8")
        legacy_summary = self.html / "graphs" / "summary" / f"{name}.json"
        legacy_summary.parent.mkdir(parents=True)
        legacy_summary.write_text("[[1, 5]]", encoding="utf-8")
        for machine, factor in zip(machines, [10, 1, 20]):
            graph = self.html / (Graph.get_file_path(machine, name) + ".json")
            graph.parent.mkdir(parents=True)
            graph.write_text(
                json.dumps(
                    [
                        [1, [factor, factor * 2, None, None]],
                        [2, None],
                        [3, [factor * 3, None, factor * 4, None]],
                    ]
                ),
                encoding="utf-8",
            )
        destination = self.root / "site-data"
        extract_site_data(self.html, destination)
        self.assertEqual(
            (destination / "graphs" / "summary" / f"{name}.json").read_text(
                encoding="utf-8"
            ),
            "[[1, 5]]",
        )
        summary = json.loads(
            (destination / "graphs" / "configurations" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(summary["series"]), 12)
        curves = {
            (
                summary["graph_parameters"][curve["graph"]]["machine"],
                summary["graph_parameters"][curve["graph"]]["python"],
                curve["parameters"]["dtype"],
                curve["parameters"]["backend"],
            ): curve["data"]
            for curve in summary["series"]
        }
        self.assertEqual(
            curves[("fast", "3.12", "'float32'", "'ort'")],
            [[1000, 1], [2000, None], [3000, 3]],
        )
        self.assertEqual(
            curves[("slow", "3.12", "'float32'", "'ort'")],
            [[1000, 10], [2000, None], [3000, 30]],
        )
        self.assertEqual(
            curves[("slow", "3.12", "'float64'", "'reference'")],
            [[1000, None], [2000, None], [3000, None]],
        )
        self.assertEqual(
            curves[("slow", "3.13", "'float32'", "'ort'")],
            [[1000, 20], [2000, None], [3000, 60]],
        )

    def test_preview_without_benchmark_parameters_preserves_scalar_values(self):
        from asv.graph import Graph

        name = "model.time_run"
        parameters = {"machine": "cpu", "onnx": None}
        index = {
            "benchmarks": {name: {"param_names": [], "params": []}},
            "graph_param_list": [parameters],
            "revision_to_date": {"1": 1000, "2": 2000},
        }
        (self.html / "index.json").write_text(json.dumps(index), encoding="utf-8")
        graph = self.html / (Graph.get_file_path(parameters, name) + ".json")
        graph.parent.mkdir(parents=True)
        graph.write_text("[[1, 2], [2, null]]", encoding="utf-8")
        write_configuration_previews(self.html)
        summary = json.loads(
            (self.html / "graphs" / "configurations" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["series"],
            [{"graph": 0, "parameters": {}, "data": [[1000, 2], [2000, None]]}],
        )
        self.assertEqual(summary["graph_parameters"], [parameters])

    def test_preview_rejects_incompatible_parameter_count(self):
        from asv.graph import Graph

        name = "model.time_run"
        parameters = {"machine": "cpu"}
        (self.html / "index.json").write_text(
            json.dumps(
                {
                    "benchmarks": {
                        name: {
                            "param_names": ["dtype"],
                            "params": [["float32", "float64"]],
                        }
                    },
                    "graph_param_list": [parameters],
                    "revision_to_date": {"1": 1000},
                }
            ),
            encoding="utf-8",
        )
        path = self.html / (Graph.get_file_path(parameters, name) + ".json")
        path.parent.mkdir(parents=True)
        path.write_text("[[1, [2]]]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parameter count"):
            write_configuration_previews(self.html)


if __name__ == "__main__":
    unittest.main()
