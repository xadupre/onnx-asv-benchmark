import json
import tempfile
import unittest
from pathlib import Path

from tools.check_results import failed_qwen_bf16_cases, successful_benchmark_count


class TestCheckResults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_results(self, results):
        machine = self.root / "machine"
        machine.mkdir()
        (machine / "result.json").write_text(
            json.dumps(
                {
                    "result_columns": ["result", "params"],
                    "results": results,
                }
            ),
            encoding="utf-8",
        )

    def test_counts_partially_successful_shard(self):
        self.write_results(
            {
                "models.dummies.mlp.MLP.time_run": [
                    [1.0, None],
                    [["'onnx-light'", "'onnx-light-cpu'"]],
                ]
            }
        )

        self.assertEqual(
            successful_benchmark_count(self.root, "models/dummies/mlp"),
            1,
        )

    def test_rejects_fully_failed_shard(self):
        self.write_results(
            {
                "models.dummies.mlp.MLP.time_run": [
                    [None, float("nan")],
                    [["'onnx-light'", "'onnx-light-cpu'"]],
                ]
            }
        )

        self.assertEqual(
            successful_benchmark_count(self.root, "models/dummies/mlp"),
            0,
        )

    def test_qwen_bf16_failure_is_not_hidden_by_other_results(self):
        params = [
            ["'Qwen/Qwen2-0.5B'"],
            ["'prefill'"],
            ["'fp32'", "'bf16'"],
            ["'onnxruntime'", "'onnx-light'", "'onnx-light-cpu'"],
        ]
        results = {}
        for method in ("time_prefill", "time_decode"):
            results[f"models.llm.qwen2.Qwen2.{method}"] = [
                [1.0, 1.0, 1.0, None, None, 1.0],
                params,
            ]
        self.write_results(results)
        self.assertEqual(
            len(failed_qwen_bf16_cases(self.root)),
            2,
        )
        for values in results.values():
            values[0][4] = 1.0
        (self.root / "machine" / "result.json").write_text(
            json.dumps({"result_columns": ["result", "params"], "results": results}),
            encoding="utf-8",
        )
        self.assertEqual(failed_qwen_bf16_cases(self.root), [])


if __name__ == "__main__":
    unittest.main()
