from benchmarks._operator import OperatorBenchmark


class GreaterOrEqual(OperatorBenchmark):
    operator = "GreaterOrEqual"
    case_name = "test_cc_greater_or_equal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
