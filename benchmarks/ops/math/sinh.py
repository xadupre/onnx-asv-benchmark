from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sinh(_OperatorBenchmark):
    operator = "Sinh"
    case_name = "test_cc_sinh_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
