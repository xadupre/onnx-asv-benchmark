from benchmarks._operator import OperatorBenchmark


class LessOrEqual(OperatorBenchmark):
    operator = "LessOrEqual"
    case_name = "test_cc_less_or_equal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
