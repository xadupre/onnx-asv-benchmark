import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asv import runner
from asv.results import Results

from tools.run_asv_capture import install_error_capture


class FailingSpawner:
    def run(self, **kwargs):
        return "NotImplemented: Add bfloat16 is unavailable", 1


class TestAsvErrorCapture(unittest.TestCase):
    def test_failed_parameter_is_saved_and_cleared_on_success(self):
        original = (runner._run_benchmark_single_param, Results.add_result, Results.save)
        self.addCleanup(setattr, runner, "_run_benchmark_single_param", original[0])
        self.addCleanup(setattr, Results, "add_result", original[1])
        self.addCleanup(setattr, Results, "save", original[2])
        install_error_capture()
        with tempfile.TemporaryDirectory() as directory:
            result = Results(
                {"machine": "machine"}, {}, "abc", 1, "3.12", "existing", {}
            )
            benchmark = {
                "name": "cpu_backend_cases.math.cases.AddBfloat16Inputs2Shard4.time_add_v14_per_channel",
                "params": [["'onnx-light-cpu'", "'onnxruntime'"]],
                "version": "v1",
            }
            failed = SimpleNamespace(
                result=[1.0, None], samples=[None, None],
                number=[None, None], stderr="RuntimeError: unavailable",
                errcode=1, profile=None,
            )
            result.add_result(benchmark, failed)
            result.save(directory)
            path = Path(directory) / result._filename
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["benchmark_errors"][
                    benchmark["name"]
                ],
                {"1": "RuntimeError: unavailable"},
            )

            benchmark["timeout"] = 60
            runner._run_benchmark_single_param(
                benchmark, FailingSpawner(), 1,
                profile=False, extra_params={}, cwd=directory,
            )
            result.add_result(benchmark, failed)
            result.save(directory)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["benchmark_errors"][
                    benchmark["name"]
                ],
                {"1": "NotImplemented: Add bfloat16 is unavailable"},
            )

            recovered = SimpleNamespace(
                result=[1.0, 2.0], samples=[None, None],
                number=[None, None], stderr="", errcode=0, profile=None,
            )
            result.add_result(benchmark, recovered)
            result.save(directory)
            self.assertNotIn(
                "benchmark_errors", json.loads(path.read_text(encoding="utf-8"))
            )


if __name__ == "__main__":
    unittest.main()
