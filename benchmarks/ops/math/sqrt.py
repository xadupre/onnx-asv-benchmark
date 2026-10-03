from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sqrt(_OperatorBenchmark):
    operator = "Sqrt"
    case_name = "test_cc_sqrt_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
