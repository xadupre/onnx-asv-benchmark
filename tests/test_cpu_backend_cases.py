import importlib
import inspect
import unittest

from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
    _all_case_names,
)
from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS
from tools.generate_cpu_backend_case_manifest import (
    build_shards,
    case_prefix,
    operator_locations,
)


class _Case:
    def __init__(self, name, operator):
        self.name = name
        self.model = _Model(operator)
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class _Model:
    def __init__(self, operator):
        self.graph = _Graph(operator)


class _Graph:
    def __init__(self, operator):
        self.node = [_Node(operator)]


class _Node:
    def __init__(self, operator):
        self.op_type = operator


class TestCpuBackendCases(unittest.TestCase):
    def test_manifest_classes_cover_distinct_cases(self):
        benchmark_types = {}
        for category, module_name, class_name, _, _, _ in CASE_SHARDS:
            module = importlib.import_module(
                f"benchmarks.cpu_backend_cases.{category}.{module_name}"
            )
            benchmark_type = getattr(module, class_name)
            self.assertTrue(inspect.isclass(benchmark_type))
            self.assertTrue(issubclass(benchmark_type, _CpuBackendCaseBenchmark))
            self.assertEqual(benchmark_type.__module__, module.__name__)
            benchmark_types[category, module_name, class_name] = benchmark_type

        self.assertEqual(len(benchmark_types), len(CASE_SHARDS))
        names = [
            name
            for benchmark_type in benchmark_types.values()
            for name in benchmark_type.params[0]
        ]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(_all_case_names()))
        self.assertTrue(
            all(
                name.startswith("test_cpu_") and name.endswith("_benchmark")
                for name in names
            )
        )
        self.assertTrue(
            all(
                len(benchmark_type.params[0]) <= 200
                for benchmark_type in benchmark_types.values()
            )
        )
        self.assertTrue(
            all(
                benchmark_type.params[1] == ("onnx-light-cpu", "onnxruntime")
                for benchmark_type in benchmark_types.values()
            )
        )

    def test_one_case_on_both_backends(self):
        module = importlib.import_module("benchmarks.cpu_backend_cases.math.abs")
        benchmark_type = module.Abs
        case_name = next(
            name for name in benchmark_type.params[0] if "_float32_" in name
        )
        for backend in benchmark_type.params[1]:
            with self.subTest(backend=backend):
                benchmark = benchmark_type()
                benchmark.setup(case_name, backend)
                benchmark.time_run(case_name, backend)
                benchmark.teardown(case_name, backend)

    def test_manifest_chunking(self):
        source = [
            _Case(f"test_cpu_abs_n{index}_float32_benchmark", "Abs")
            for index in range(5)
        ]
        shards = build_shards(
            source,
            chunk_size=2,
            locations={"Abs": ("math", "abs")},
        )
        self.assertEqual(
            shards,
            [
                ("math", "abs", "AbsPart01", "abs", 0, 2),
                ("math", "abs", "AbsPart02", "abs", 2, 4),
                ("math", "abs", "AbsPart03", "abs", 4, None),
            ],
        )
        self.assertTrue(source[0].unloaded)

    def test_operator_locations_match_ops_hierarchy(self):
        locations = operator_locations("benchmarks/ops")
        self.assertEqual(locations["Abs"], ("math", "abs"))
        self.assertEqual(
            locations["BatchNormalization"],
            ("nn", "batch_normalization"),
        )
        self.assertEqual(
            locations["GroupQueryAttention"],
            ("nn", "group_query_attention"),
        )

    def test_case_prefix_rejects_unexpected_names(self):
        self.assertEqual(case_prefix("test_cpu_abs_float32_benchmark"), "abs")
        with self.assertRaises(ValueError):
            case_prefix("test_cc_abs_float32_benchmark")
