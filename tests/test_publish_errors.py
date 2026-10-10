import json
import tempfile
import unittest
from pathlib import Path

from asv.graph import Graph

from tools.publish_errors import publish_errors


class TestPublishErrors(unittest.TestCase):
    def test_errors_match_graph_machine_revision_and_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html = root / "html"
            results = root / "results"
            html.mkdir()
            (results / "machine").mkdir(parents=True)
            benchmark = "cpu_backend_cases.math.cases.AddBfloat16Inputs2Shard4.time_add_v14_per_channel"
            graph_params = {
                "branch": "main",
                "machine": "Example CPU (8 vCPU)",
                "onnxruntime": "1.30",
            }
            graph_path = Graph.get_file_path(graph_params, benchmark) + ".json"
            target = html / graph_path
            target.parent.mkdir(parents=True)
            target.write_text("[]", encoding="utf-8")
            (html / "index.json").write_text(
                json.dumps({
                    "revision_to_hash": {"102": "abc"},
                    "graph_param_list": [graph_params],
                    "benchmarks": {benchmark: {"params": [["'onnx-light-cpu'", "'onnxruntime'"]]}},
                }),
                encoding="utf-8",
            )
            (results / "machine" / "run.json").write_text(
                json.dumps({
                    "commit_hash": "abc",
                    "params": {
                        "machine": "Example CPU (8 vCPU)",
                        "onnxruntime": "1.30",
                    },
                    "env_vars": {},
                    "result_columns": ["result", "params"],
                    "results": {
                        benchmark: [None, [["'onnxruntime'"]]]
                    },
                    "benchmark_errors": {benchmark: {"0": "NotImplemented: Add bfloat16"}},
                }),
                encoding="utf-8",
            )
            publish_errors(results, html)
            published = json.loads((html / "errors" / (benchmark + ".json")).read_text())
            self.assertEqual(
                published[graph_path], {"102": {"1": "NotImplemented: Add bfloat16"}}
            )
            self.assertEqual(
                json.loads((html / "index.json").read_text())["error_benchmarks"],
                [benchmark],
            )


if __name__ == "__main__":
    unittest.main()
