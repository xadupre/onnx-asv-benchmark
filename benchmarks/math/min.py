from benchmarks._operator import OperatorBenchmark


class Min(OperatorBenchmark):
    operator = "Min"
    case_name = "test_cc_min_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
