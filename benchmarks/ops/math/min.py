from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Min(_OperatorBenchmark):
    operator = "Min"
    case_name = "test_cc_min_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
