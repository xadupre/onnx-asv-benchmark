import importlib
import inspect
import unittest

from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
    _all_case_records,
)
from benchmarks.cpu_backend_cases._manifest import CASE_SHARDS
from tools.generate_cpu_backend_case_manifest import (
    build_shards,
    case_metadata,
    case_prefix,
    class_name,
    operator_locations,
    simplified_case_name,
)


class _Case:
    def __init__(self, name, operator):
        self.name = name
        self.model = _Model(operator)
        self.data_sets = [_DataSet()]
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class _Model:
    def __init__(self, operator):
        self.graph = _Graph(operator)


class _DataSet:
    def __init__(self):
        self.inputs = [_Tensor()]


class _Tensor:
    data_type = 1
    shape = (1,)


class _Graph:
    def __init__(self, operator):
        self.node = [_Node(operator)]


class _Node:
    def __init__(self, operator):
        self.op_type = operator


class TestCpuBackendCases(unittest.TestCase):
    def test_manifest_classes_cover_distinct_cases(self):
        benchmark_types = {}
        for category, class_name, _, _ in CASE_SHARDS:
            module = importlib.import_module(
                f"benchmarks.cpu_backend_cases.{category}.cases"
            )
            benchmark_type = getattr(module, class_name)
            self.assertTrue(inspect.isclass(benchmark_type))
            self.assertTrue(issubclass(benchmark_type, _CpuBackendCaseBenchmark))
            self.assertEqual(benchmark_type.__module__, module.__name__)
            benchmark_types[category, class_name] = benchmark_type

        self.assertEqual(len(benchmark_types), len(CASE_SHARDS))
        covered = {
            record[0]
            for benchmark_type in benchmark_types.values()
            for record in _all_case_records()
            if record[0].startswith(
                f"test_cpu_{benchmark_type.case_prefix}_"
            )
            and record[2] == benchmark_type.case_dtypes
        }
        self.assertEqual(covered, {record[0] for record in _all_case_records()})

    def test_one_case_on_both_backends(self):
        module = importlib.import_module("benchmarks.cpu_backend_cases.math.cases")
        benchmark_type = module.AbsFloat32
        benchmark_name = next(
            name for name in dir(benchmark_type) if name.startswith("time_")
        )
        method = getattr(benchmark_type, benchmark_name)
        for backend in method.params[-1]:
            with self.subTest(backend=backend):
                benchmark = benchmark_type()
                parameters = tuple(axis[0] for axis in method.params[:-1]) + (backend,)
                method.setup(*parameters)
                getattr(benchmark, benchmark_name)(*parameters)
                method.teardown(*parameters)

    def test_manifest_chunking(self):
        source = [
            _Case(f"test_cpu_abs_n{index}_float32_benchmark", "Abs")
            for index in range(5)
        ]
        shards = build_shards(source, locations={"Abs": ("math", "abs")})
        self.assertEqual(
            shards,
            [
                ("math", "abs", "AbsFloat32", "abs", ("float32",)),
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

    def test_simplified_case_name(self):
        self.assertEqual(
            simplified_case_name(
                "test_cpu_sub_v14_row_float64xfloat64_to_float64_"
                "swapped_n1048576_benchmark",
                ("float64", "float64"),
            ),
            "sub_v14_row",
        )
        self.assertEqual(
            class_name("LinearAttention", ("bfloat16",) * 6),
            "LinearAttentionBfloat16Inputs6",
        )
