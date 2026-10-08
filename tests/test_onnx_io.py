import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import onnx
import onnx_light.onnx as onnxl

from benchmarks.builder._onnx_io import (
    LOAD_CASES,
    PARSE_CASES,
    SAVE_CASES,
    SAVE_CPP_CASES,
    SERIALIZE_CASES,
    case_parameters,
)
from benchmarks.builder.load import onnx_io as load_onnx_io
from benchmarks.builder.load.onnx_io import (
    OnnxLoad,
    OnnxReferenceEvaluator,
)
from benchmarks.builder.parse.onnx_io import OnnxParse
from benchmarks.builder.save.onnx_io import OnnxSave, OnnxSaveCpp
from benchmarks.builder.serialize.onnx_io import OnnxSerialize


class TestOnnxIO(unittest.TestCase):
    def test_cases(self):
        cases = (
            *LOAD_CASES,
            *SAVE_CASES,
            *SERIALIZE_CASES,
            *PARSE_CASES,
            *SAVE_CPP_CASES,
        )
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(set(cases)), 33)
        self.assertFalse(hasattr(load_onnx_io, "OnnxLoadCpp"))
        for benchmark in (
            OnnxLoad,
            OnnxSave,
            OnnxSaveCpp,
            OnnxSerialize,
            OnnxParse,
        ):
            self.assertEqual(
                benchmark.param_names, ("shape", "dtype", "case", "library")
            )
            self.assertEqual(benchmark.params[1], ("float32",))
            self.assertEqual(benchmark.params[0], ("X=dynamicx2048 (40 Gemm)",))
            self.assertEqual(benchmark.params[2:], case_parameters(benchmark.cases))
            self.assertTrue(all(value.count("/") == 1 for value in benchmark.params[2]))
            self.assertTrue(all("/" not in value for value in benchmark.params[3]))

    def test_reference_evaluator_creation(self):
        class SmallBenchmark(OnnxReferenceEvaluator):
            n_init = 2
            dim = 8

        benchmark = SmallBenchmark()
        parameters = ("X=dynamicx8 (2 Gemm)", "float32")
        benchmark.setup(*parameters)
        try:
            self.assertIsNone(benchmark.session)
            benchmark.time_run(*parameters)
            self.assertIsNotNone(benchmark.session)
            self.assertIs(benchmark.session._model, benchmark.model)
        finally:
            benchmark.teardown(*parameters)
        self.assertIsNone(benchmark.model)
        self.assertIsNone(benchmark.session)

    def test_python_cases(self):
        for benchmark_type in (OnnxLoad, OnnxSave, OnnxSerialize, OnnxParse):

            class SmallBenchmark(benchmark_type):
                n_init = 2
                dim = 8

            for case in benchmark_type.params[2]:
                for library in benchmark_type.params[3]:
                    full_case = f"{case}/{library}"
                    if full_case not in benchmark_type.cases:
                        continue
                    if (
                        library == "ir-py"
                        and importlib.util.find_spec("onnx_ir") is None
                    ):
                        continue
                    with self.subTest(case=case, library=library):
                        benchmark = SmallBenchmark()
                        parameters = (
                            "X=dynamicx8 (2 Gemm)",
                            "float32",
                            case,
                            library,
                        )
                        try:
                            benchmark.setup(*parameters)
                            benchmark.time_run(*parameters)
                            if case.startswith("load/"):
                                self.assertIsNotNone(benchmark.loaded_model)
                            if case.startswith("save/"):
                                self.assertTrue(Path(benchmark.out).is_file())
                                if "/2file" in case:
                                    self.assertTrue(Path(benchmark.out_data).is_file())
                                    self.assertEqual(
                                        len(onnx.load(benchmark.out).graph.initializer),
                                        2,
                                    )
                            if case.startswith("load/2file"):
                                self.assertEqual(
                                    len(
                                        onnxl.load(
                                            benchmark.external_path
                                        ).graph.initializer
                                    ),
                                    2,
                                )
                        finally:
                            if hasattr(benchmark, "tmp"):
                                benchmark.teardown(*parameters)
                        if case.startswith("load/"):
                            self.assertIsNone(benchmark.loaded_model)

    def test_unsupported_case_library_combination(self):
        benchmark = OnnxLoad()
        parameters = (
            "X=dynamicx2048 (40 Gemm)",
            "float32",
            "load/1filex4",
            "onnx",
        )
        with self.assertRaises(NotImplementedError):
            benchmark.setup(*parameters)

    def test_cpp_cases_skip_without_executables(self):
        old_ci = os.environ.get("CI")
        old_cicpp = os.environ.get("CICPP")
        os.environ["CI"] = "1"
        os.environ.pop("CICPP", None)
        try:
            benchmark = OnnxSaveCpp()
            parameters = (
                "X=dynamicx2048 (40 Gemm)",
                "float32",
                "save/1filex1",
                "onnxlight-cpp",
            )
            with self.assertRaises(NotImplementedError):
                benchmark.setup(*parameters)
            benchmark.teardown(*parameters)
        finally:
            if old_ci is None:
                os.environ.pop("CI", None)
            else:
                os.environ["CI"] = old_ci
            if old_cicpp is None:
                os.environ.pop("CICPP", None)
            else:
                os.environ["CICPP"] = old_cicpp

    def test_cpp_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "save_onnx_light_time"
            executable.write_text(
                "#!/bin/sh\n"
                "echo 'Average save (ms): 2'\n"
                "echo 'Median save (ms): 2'\n"
                "echo 'Min save (ms): 2'\n"
                "echo 'Max save (ms): 2'\n"
            )
            executable.chmod(0o755)
            old_path = os.environ.get("PATH", "")
            old_cicpp = os.environ.get("CICPP")
            os.environ["PATH"] = directory + os.pathsep + old_path
            os.environ["CICPP"] = "1"
            try:
                class SmallCpp(OnnxSaveCpp):
                    n_init = 2
                    dim = 8

                for full_case in OnnxSaveCpp.cases:
                    case, library = full_case.rsplit("/", 1)
                    with self.subTest(case=case, library=library):
                        benchmark = SmallCpp()
                        parameters = (
                            "X=dynamicx8 (2 Gemm)",
                            "float32",
                            case,
                            library,
                        )
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
