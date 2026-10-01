import importlib
import inspect
import pkgutil
import unittest
from pathlib import Path

from benchmarks._operator import (
    OperatorBenchmark,
    QuantizePagedCacheBenchmark,
)
from benchmarks.models.matmul_add import MatMulAdd
from benchmarks.models.mlp import MLP
from benchmarks.models.tiny_llm import TinyLLM

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


def operator_benchmarks():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    benchmarks = {}
    for category in OPERATOR_COUNTS:
        package = importlib.import_module(f"benchmarks.{category}")
        category_benchmarks = {}
        for module_info in pkgutil.iter_modules(
            package.__path__,
            f"{package.__name__}.",
        ):
            module = importlib.import_module(module_info.name)
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
            raise AssertionError(f"Missing package benchmarks.{category}.")
    return benchmarks


class TestBenchmarks(unittest.TestCase):
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
                    <= {"onnxruntime", "onnx-reference", "onnx-light"}
                )
        self.assertEqual(len(operators), 225)

    def test_one_operator_per_category(self):
        benchmarks = operator_benchmarks()
        for category, operator in SMOKE_TESTS.items():
            with self.subTest(category=category, operator=operator):
                benchmark = benchmarks[category][operator]()
                benchmark.setup("onnx-light")
                benchmark.time_run("onnx-light")

    def test_model_benchmarks(self):
        for benchmark_type in (MatMulAdd, MLP, TinyLLM):
            for backend in benchmark_type.params:
                with self.subTest(
                    benchmark=benchmark_type.__name__,
                    backend=backend,
                ):
                    benchmark = benchmark_type()
                    benchmark.setup(backend)
                    benchmark.time_run(backend)
                    teardown = getattr(benchmark, "teardown", None)
                    if teardown is not None:
                        teardown(backend)


if __name__ == "__main__":
    unittest.main()
