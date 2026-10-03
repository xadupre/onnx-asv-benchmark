from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Celu(_OperatorBenchmark):
    operator = "Celu"
    case_name = "test_cc_celu_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32',)
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
