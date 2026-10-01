from benchmarks._operator import OperatorBenchmark


class Sum(OperatorBenchmark):
    operator = "Sum"
    case_name = "test_cc_sum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
