import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import onnx
import onnx_light.onnx as onnxl
from benchmarks.models.dummies.onnx_io import (
    BYTE_CASES,
    CPP_CASES,
    LOAD_CASES,
    SAVE_CASES,
    OnnxBytes,
    OnnxCpp,
    OnnxLoad,
    OnnxSave,
)


class TestOnnxIO(unittest.TestCase):
    def test_cases(self):
        cases = (*LOAD_CASES, *SAVE_CASES, *BYTE_CASES, *CPP_CASES)
        self.assertEqual(len(cases), 42)
        self.assertEqual(len(set(cases)), 42)
        for benchmark in (OnnxLoad, OnnxSave, OnnxBytes, OnnxCpp):
            self.assertEqual(benchmark.param_names, ("shape", "dtype", "case"))
            self.assertEqual(benchmark.params[1], ("float32",))
            self.assertEqual(benchmark.params[0], ("X=dynamicx2048 (40 Gemm)",))

    def test_python_cases(self):
        for benchmark_type in (OnnxLoad, OnnxSave, OnnxBytes):
            class SmallBenchmark(benchmark_type):
                n_init = 2
                dim = 8

            for case in benchmark_type.params[2]:
                if "/ir-py" in case and importlib.util.find_spec("onnx_ir") is None:
                    continue
                with self.subTest(case=case):
                    benchmark = SmallBenchmark()
                    parameters = ("X=dynamicx8 (2 Gemm)", "float32", case)
                    try:
                        benchmark.setup(*parameters)
                        benchmark.time_run(*parameters)
                        if case.startswith("save/"):
                            self.assertTrue(Path(benchmark.out).is_file())
                            if "/2file" in case:
                                self.assertTrue(Path(benchmark.out_data).is_file())
                                self.assertEqual(
                                    len(onnx.load(benchmark.out).graph.initializer), 2
                                )
                        if case.startswith("load/2file"):
                            self.assertEqual(
                                len(onnxl.load(benchmark.external_path).graph.initializer),
                                2,
                            )
                    finally:
                        if hasattr(benchmark, "tmp"):
                            benchmark.teardown(*parameters)

    def test_cpp_cases(self):
        class SmallCpp(OnnxCpp):
            n_init = 2
            dim = 8

        with tempfile.TemporaryDirectory() as directory:
            for name in ("load_onnx_time", "load_onnx_light_time", "save_onnx_light_time"):
                operation = name.split("_", 1)[0]
                executable = Path(directory) / name
                executable.write_text(
                    "#!/bin/sh\n"
                    f"echo 'Average {operation} (ms): 2'\n"
                    f"echo 'Median {operation} (ms): 2'\n"
                    f"echo 'Min {operation} (ms): 2'\n"
                    f"echo 'Max {operation} (ms): 2'\n"
                )
                executable.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            old_cicpp = os.environ.get("CICPP")
            os.environ["PATH"] = directory + os.pathsep + old_path
            os.environ["CICPP"] = "1"
            try:
                for case in CPP_CASES:
                    with self.subTest(case=case):
                        benchmark = SmallCpp()
                        parameters = ("X=dynamicx8 (2 Gemm)", "float32", case)
                        benchmark.setup(*parameters)
                        try:
                            self.assertEqual(benchmark.track_run(*parameters), 0.002)
                        finally:
                            benchmark.teardown(*parameters)
            finally:
                os.environ["PATH"] = old_path
                if old_cicpp is None:
                    os.environ.pop("CICPP", None)
                else:
                    os.environ["CICPP"] = old_cicpp


if __name__ == "__main__":
    unittest.main()
