from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LpNormalization(_OperatorBenchmark):
    operator = "LpNormalization"
    case_name = "test_cc_lpnormalization_default_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
