import json
import tempfile
import unittest
from pathlib import Path

from tools.check_results import successful_benchmark_count


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


if __name__ == "__main__":
    unittest.main()
