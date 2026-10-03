from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Softplus(_OperatorBenchmark):
    operator = "Softplus"
    case_name = "test_cc_softplus_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
