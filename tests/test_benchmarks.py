import importlib
import inspect
import pkgutil
import unittest
from pathlib import Path

import onnx_light_cpu
from benchmarks._operator import (
    OperatorBenchmark,
    QuantizePagedCacheBenchmark,
)
from benchmarks.common import MODEL_DTYPES
from benchmarks.models.dummies.matmul_add import MatMulAdd
from benchmarks.models.dummies.mlp import MLP
from benchmarks.models.llm.tiny_llm import PRECISIONS, TinyLLM, TinyLLMGenAI
from onnx_light_cpu import (
    registered_kernel_names,
    set_kernel_usage_recording,
    used_kernel_names,
)

OPERATOR_COUNTS = {
    "generator": 10,
    "image": 1,
    "logical": 17,
    "math": 70,
    "nn": 31,
    "object_detection": 2,
    "optional": 3,
    "preview": 1,
    "quantization": 7,
    "reduction": 12,
    "rt": 1,
    "sequence": 8,
    "tensor": 36,
    "text": 5,
    "traditionalml": 18,
    "training": 3,
}

SMOKE_TESTS = {
    "generator": "Constant",
    "image": "ImageDecoder",
    "logical": "And",
    "math": "Abs",
    "nn": "AveragePool",
    "object_detection": "NonMaxSuppression",
    "optional": "OptionalHasElement",
    "preview": "FlexAttention",
    "quantization": "QuantizeLinear",
    "reduction": "ReduceSum",
    "rt": "DelayedInitializer",
    "sequence": "SequenceConstruct",
    "tensor": "Identity",
    "text": "StringConcat",
    "traditionalml": "Binarizer",
    "training": "Momentum",
}


def requires_onnx_light_cpu(minimum_version):
    current = tuple(int(part) for part in onnx_light_cpu.__version__.split("."))
    minimum = tuple(int(part) for part in minimum_version.split("."))
    return unittest.skipUnless(
        current >= minimum,
        f"onnx-light-cpu>={minimum_version} is required; found "
        f"{onnx_light_cpu.__version__}.",
    )


def operator_benchmarks():
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "ops"
    benchmarks = {}
    for category in OPERATOR_COUNTS:
        package = importlib.import_module(f"benchmarks.ops.{category}")
        category_benchmarks = {}
        for module_info in pkgutil.iter_modules(
            package.__path__,
            f"{package.__name__}.",
        ):
            module = importlib.import_module(module_info.name)
            public_imports = [
                name
                for name, value in inspect.getmembers(module, inspect.isclass)
                if not name.startswith("_")
                and value.__module__ != module.__name__
                and issubclass(
                    value,
                    (OperatorBenchmark, QuantizePagedCacheBenchmark),
                )
            ]
            if public_imports:
                raise AssertionError(
                    f"{module_info.name} publicly imports benchmark classes "
                    f"{public_imports}, which ASV would discover twice."
                )
            classes = [
                value
                for _, value in inspect.getmembers(module, inspect.isclass)
                if value.__module__ == module.__name__
                and issubclass(
                    value,
                    (OperatorBenchmark, QuantizePagedCacheBenchmark),
                )
            ]
            if len(classes) != 1:
                raise AssertionError(
                    f"{module_info.name} defines {len(classes)} benchmarks."
                )
            category_benchmarks[classes[0].__name__] = classes[0]
        benchmarks[category] = category_benchmarks
        if not (root / category / "__init__.py").exists():
            raise AssertionError(f"Missing package benchmarks.ops.{category}.")
    return benchmarks


class TestBenchmarks(unittest.TestCase):
    def run_benchmark(self, benchmark_type, parameter_values):
        benchmark = benchmark_type()
        benchmark.setup(*parameter_values)
        time_methods = [
            getattr(benchmark, name)
            for name in dir(benchmark)
            if name.startswith("time_")
        ]
        self.assertTrue(time_methods)
        for time_method in time_methods:
            time_method(*parameter_values)
        teardown = getattr(benchmark, "teardown", None)
        if teardown is not None:
            teardown(*parameter_values)

    def test_operator_coverage(self):
        benchmarks = operator_benchmarks()
        operators = set()
        for category, expected_count in OPERATOR_COUNTS.items():
            category_operators = set(benchmarks[category])
            self.assertEqual(len(category_operators), expected_count)
            self.assertTrue(operators.isdisjoint(category_operators))
            operators.update(category_operators)
            for benchmark in benchmarks[category].values():
                self.assertIn("onnx-light", benchmark.params)
                self.assertTrue(
                    set(benchmark.params)
                    <= {
                        "onnxruntime",
                        "onnx-reference",
                        "onnx-light",
                        "onnx-light-cpu",
                    }
                )
                self.assertIn("onnx-light-cpu", benchmark.params)
        self.assertEqual(len(operators), 225)

    def test_one_operator_per_category(self):
        benchmarks = operator_benchmarks()
        for category, operator in SMOKE_TESTS.items():
            with self.subTest(category=category, operator=operator):
                benchmark = benchmarks[category][operator]()
                benchmark.setup("onnx-light")
                benchmark.time_run("onnx-light")

    # 0.1.16 omitted the compiled _cpuregister extension; see onnx-light-cpu#827.
    @requires_onnx_light_cpu("0.1.17")
    def test_onnx_light_cpu_operator(self):
        benchmark = operator_benchmarks()["math"]["Add"]()
        benchmark.setup("onnx-light-cpu")
        set_kernel_usage_recording(benchmark.session, True)
        benchmark.time_run("onnx-light-cpu")
        self.assertIn(
            registered_kernel_names()["Add"],
            used_kernel_names(benchmark.session),
        )

    def test_model_benchmarks(self):
        self.assertEqual(MatMulAdd.params[0], MODEL_DTYPES)
        self.assertEqual(MLP.params[0], MODEL_DTYPES)
        for benchmark_type in (MatMulAdd, MLP, TinyLLM, TinyLLMGenAI):
            self.assertEqual(benchmark_type.param_names[0], "dtype")
        self.assertEqual(TinyLLM.params[0], PRECISIONS)
        self.assertEqual(TinyLLMGenAI.params[0], PRECISIONS)
        self.assertEqual(
            TinyLLMGenAI.params[1],
            ("onnxruntime-genai", "onnx-light", "onnx-light-cpu"),
        )
        for benchmark_type in (MatMulAdd, MLP, TinyLLM, TinyLLMGenAI):
            params = benchmark_type.params
            dtype_values, backend_values = params
            params = (
                (dtype_name, backend)
                for dtype_name in dtype_values
                for backend in backend_values
            )
            for parameter_values in params:
                backend = parameter_values[-1]
                if backend == "onnx-light-cpu" or (
                    benchmark_type is TinyLLMGenAI and backend == "onnx-light"
                ):
                    continue
                is_available = getattr(benchmark_type, "is_available", None)
                if is_available is not None and not is_available(*parameter_values):
                    benchmark = benchmark_type()
                    with self.assertRaises(NotImplementedError):
                        benchmark.setup(*parameter_values)
                    benchmark.teardown(*parameter_values)
                    continue
                with self.subTest(
                    benchmark=benchmark_type.__name__,
                    parameters=parameter_values,
                ):
                    self.run_benchmark(benchmark_type, parameter_values)

    # Generation exposes the missing 0.1.16 registration extension and kernels.
    @requires_onnx_light_cpu("0.1.17")
    def test_onnx_light_cpu_models(self):
        for benchmark_type, parameter_values in (
            (MatMulAdd, ("float64", "onnx-light-cpu")),
            (MLP, ("float64", "onnx-light-cpu")),
            *(
                (TinyLLMGenAI, (precision, backend))
                for precision in PRECISIONS
                for backend in ("onnx-light", "onnx-light-cpu")
            ),
        ):
            with self.subTest(
                benchmark=benchmark_type.__name__,
                parameters=parameter_values,
            ):
                self.run_benchmark(benchmark_type, parameter_values)


if __name__ == "__main__":
    unittest.main()
