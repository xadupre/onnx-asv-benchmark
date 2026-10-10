import importlib
import inspect
import itertools
import pkgutil
import unittest
from pathlib import Path
import numpy as np
import onnx_light
import onnx_light_cpu
from benchmarks._operator import (
    OperatorBenchmark,
    QuantizePagedCacheBenchmark,
)
from benchmarks.common import MODEL_DTYPES
from benchmarks.models.dummies.matmul_add import MatMulAdd
from benchmarks.models.dummies.mlp import MLP
from benchmarks.models.llm.qwen2 import Qwen2, Qwen2GenAI
from benchmarks.models.llm.tiny_llm import (
    CACHE_LENGTH,
    PRECISIONS,
    TinyLLM,
    TinyLLMGenAI,
    _cache_dtype,
)
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


def requires_onnx_light(minimum_version):
    current = tuple(int(part) for part in onnx_light.__version__.split("."))
    minimum = tuple(int(part) for part in minimum_version.split("."))
    return unittest.skipUnless(
        current >= minimum,
        f"onnx-light>={minimum_version} is required; found {onnx_light.__version__}.",
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
                if hasattr(benchmark, "dtypes"):
                    self.assertEqual(
                        benchmark.param_names,
                        ("shape", "dtype", "backend"),
                    )
                    self.assertEqual(benchmark.params[1], benchmark.dtypes)
                    backend_index = 2
                else:
                    self.assertEqual(benchmark.param_names, ("shape", "backend"))
                    backend_index = 1
                self.assertEqual(len(benchmark.params[0]), 1)
                self.assertTrue(benchmark.params[0][0])
                self.assertEqual(
                    benchmark.params[backend_index],
                    (
                        "onnxruntime",
                        "onnx-reference",
                        "onnx-light",
                        "onnx-light-cpu",
                    ),
                )
        self.assertEqual(len(operators), 225)

    def test_one_operator_per_category(self):
        benchmarks = operator_benchmarks()
        for category, operator in SMOKE_TESTS.items():
            with self.subTest(category=category, operator=operator):
                benchmark = benchmarks[category][operator]()
                shape = benchmark.params[0][0]
                parameters = (
                    (shape, benchmark.dtypes[0], "onnx-light")
                    if hasattr(benchmark, "dtypes")
                    else (shape, "onnx-light")
                )
                benchmark.setup(*parameters)
                benchmark.time_run(*parameters)

    @requires_onnx_light_cpu("0.1.18")
    def test_onnx_light_cpu_operator(self):
        benchmark = operator_benchmarks()["math"]["Add"]()
        shape = benchmark.params[0][0]
        benchmark.setup(shape, "float32", "onnx-light-cpu")
        set_kernel_usage_recording(benchmark.session, True)
        benchmark.time_run(shape, "float32", "onnx-light-cpu")
        self.assertIn(
            registered_kernel_names()["Add"],
            used_kernel_names(benchmark.session),
        )

    def test_operator_dtypes(self):
        benchmarks = operator_benchmarks()
        expected = {
            ("logical", "Equal"): (
                "float16",
                "float64",
                "bfloat16",
                "uint8",
                "int64",
            ),
            ("logical", "IsNaN"): ("float16", "float64", "bfloat16"),
            ("math", "Add"): ("float16", "float64", "bfloat16", "uint8", "int64"),
            ("math", "Div"): ("float16", "float64", "bfloat16", "uint8", "int64"),
            ("math", "Gemm"): (
                "float16",
                "float64",
                "bfloat16",
                "uint32",
                "int64",
            ),
            ("math", "MatMul"): (
                "float16",
                "float64",
                "bfloat16",
                "uint32",
                "int64",
            ),
            ("tensor", "Gather"): (
                "float16",
                "float64",
                "bfloat16",
                "uint8",
                "int64",
            ),
            ("tensor", "ScatterElements"): (
                "float16",
                "float64",
                "bfloat16",
                "uint8",
                "int64",
            ),
        }
        for (category, operator), expected_dtypes in expected.items():
            benchmark_type = benchmarks[category][operator]
            with self.subTest(operator=operator):
                self.assertEqual(benchmark_type.dtypes[0], "float32")
                for dtype in expected_dtypes:
                    self.assertIn(dtype, benchmark_type.dtypes)

        for category, operator, dtype in (
            ("logical", "Equal", "int32"),
            ("logical", "IsNaN", "float16"),
            ("math", "Add", "int32"),
            ("math", "Div", "float16"),
            ("math", "MatMul", "int32"),
            ("tensor", "Gather", "int32"),
            ("tensor", "ScatterElements", "int32"),
        ):
            with self.subTest(operator=operator, dtype=dtype):
                benchmark_type = benchmarks[category][operator]
                self.run_benchmark(
                    benchmark_type,
                    (benchmark_type.params[0][0], dtype, "onnx-reference"),
                )

    def test_model_benchmarks(self):
        self.assertEqual(MatMulAdd.params[1], MODEL_DTYPES)
        self.assertEqual(MLP.params[1], MODEL_DTYPES)
        for benchmark_type in (MatMulAdd, MLP, TinyLLM, TinyLLMGenAI):
            self.assertEqual(
                benchmark_type.param_names,
                ("shape", "dtype", "backend"),
            )
            self.assertEqual(len(benchmark_type.params[0]), 1)
            self.assertTrue(benchmark_type.params[0][0])
        for benchmark_type in (Qwen2, Qwen2GenAI):
            self.assertEqual(
                benchmark_type.param_names,
                ("model", "shape", "dtype", "backend"),
            )
            self.assertEqual(benchmark_type.params[0], ("Qwen/Qwen2-0.5B",))
            self.assertEqual(len(benchmark_type.params[1]), 1)
            self.assertTrue(benchmark_type.params[1][0])
        self.assertEqual(Qwen2.time_prefill.pretty_name, "Qwen2-0.5B prefill")
        self.assertEqual(Qwen2.time_decode.pretty_name, "Qwen2-0.5B decode")
        self.assertEqual(
            Qwen2GenAI.time_generate.pretty_name,
            "Qwen2-0.5B generation",
        )
        for inference_type, generation_type in (
            (Qwen2, Qwen2GenAI),
            (TinyLLM, TinyLLMGenAI),
        ):
            dtype_index = inference_type.param_names.index("dtype")
            backend_index = inference_type.param_names.index("backend")
            self.assertEqual(inference_type.params[dtype_index], PRECISIONS)
            self.assertEqual(
                inference_type.params[backend_index],
                ("onnxruntime", "onnx-reference", "onnx-light", "onnx-light-cpu"),
            )
            dtype_index = generation_type.param_names.index("dtype")
            backend_index = generation_type.param_names.index("backend")
            self.assertEqual(generation_type.params[dtype_index], PRECISIONS)
            self.assertEqual(
                generation_type.params[backend_index],
                (
                    "onnxruntime-genai",
                    "onnx-reference",
                    "onnx-light",
                    "onnx-light-cpu",
                ),
            )
        for benchmark_type in (
            MatMulAdd,
            MLP,
            Qwen2,
            Qwen2GenAI,
            TinyLLM,
            TinyLLMGenAI,
        ):
            for parameter_values in itertools.product(*benchmark_type.params):
                backend = parameter_values[-1]
                if backend == "onnx-light-cpu" or (
                    benchmark_type in {Qwen2, Qwen2GenAI, TinyLLM, TinyLLMGenAI}
                    and backend in {"onnx-reference", "onnx-light"}
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

    @requires_onnx_light("0.1.31")
    @requires_onnx_light_cpu("0.1.18")
    def test_onnx_light_cpu_models(self):
        for benchmark_type, parameter_values in (
            *(
                (
                    benchmark_type,
                    (benchmark_type.params[0][0], dtype_name, "onnx-light-cpu"),
                )
                for benchmark_type in (MatMulAdd, MLP)
                for dtype_name in MODEL_DTYPES
            ),
            *(
                (
                    benchmark_type,
                    (
                        *(values[0] for values in benchmark_type.params[:-2]),
                        precision,
                        backend,
                    ),
                )
                for benchmark_type in (Qwen2GenAI, TinyLLMGenAI)
                for precision in PRECISIONS
                for backend in ("onnx-light", "onnx-light-cpu")
            ),
        ):
            with self.subTest(
                benchmark=benchmark_type.__name__,
                parameters=parameter_values,
            ):
                self.run_benchmark(benchmark_type, parameter_values)

    def test_qwen_inference_input_contracts(self):
        outputs = ["logits", "present.0.key", "present.0.value"]
        base_inputs = [
            "input_ids",
            "attention_mask",
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]
        self_test = self

        class Session:
            def __init__(self, dtype, logits_dtype):
                self.dtype = dtype
                self.logits_dtype = logits_dtype

            def run(self, output_names, feeds):
                length = feeds["input_ids"].shape[1]
                past_length = feeds["past_key_values.0.key"].shape[2]
                self_test.assertEqual(
                    feeds["attention_mask"].shape, (1, past_length + length)
                )
                if "position_ids" in feeds:
                    np.testing.assert_array_equal(
                        feeds["position_ids"],
                        np.arange(past_length, past_length + length).reshape(1, -1),
                    )
                cache = np.zeros((1, 4, past_length + length, 64), dtype=self.dtype)
                return [
                    np.zeros((1, length, 32000), dtype=self.logits_dtype),
                    cache,
                    cache.copy(),
                ]

        config = Qwen2.make_config()
        for dtype in ("fp32", "bf16"):
            for has_position_ids in (False, True):
                with self.subTest(dtype=dtype, position_ids=has_position_ids):
                    benchmark = Qwen2()
                    benchmark.session = Session(
                        _cache_dtype(dtype),
                        np.float32 if dtype == "bf16" else _cache_dtype(dtype),
                    )
                    inputs = base_inputs + (["position_ids"] if has_position_ids else [])
                    benchmark._setup_inference(inputs, outputs, dtype, config)
                    self.assertEqual(
                        "position_ids" in benchmark.prefill_feeds, has_position_ids
                    )
                    self.assertEqual(
                        "position_ids" in benchmark.decode_feeds, has_position_ids
                    )
                    self.assertEqual(
                        benchmark.decode_feeds["past_key_values.0.key"].shape,
                        (1, 4, CACHE_LENGTH, 64),
                    )
                    benchmark.time_prefill(
                        Qwen2.params[0][0], Qwen2.params[1][0], dtype, "onnx-light"
                    )
                    benchmark.time_decode(
                        Qwen2.params[0][0], Qwen2.params[1][0], dtype, "onnx-light"
                    )
        benchmark = Qwen2()
        benchmark.session = Session(np.float32, np.float32)
        with self.assertRaisesRegex(ValueError, "unexpected_input"):
            benchmark._setup_inference(
                base_inputs + ["unexpected_input"], outputs, "fp32", config
            )
        with self.assertRaisesRegex(ValueError, "present.0.value"):
            benchmark._setup_inference(base_inputs, outputs[:-1], "fp32", config)

        class BadSession(Session):
            def run(self, output_names, feeds):
                values = super().run(output_names, feeds)
                values[1] = values[1].astype(np.float16)
                return values

        benchmark.session = BadSession(np.float32, np.float32)
        with self.assertRaisesRegex(AssertionError, "present.0.key"):
            benchmark._setup_inference(base_inputs, outputs, "fp32", config)


if __name__ == "__main__":
    unittest.main()
