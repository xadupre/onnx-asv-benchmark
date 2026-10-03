from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Asin(_OperatorBenchmark):
    operator = "Asin"
    case_name = "test_cc_asin_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
