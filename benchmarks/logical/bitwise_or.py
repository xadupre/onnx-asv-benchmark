from benchmarks._operator import OperatorBenchmark


class BitwiseOr(OperatorBenchmark):
    operator = "BitwiseOr"
    case_name = "test_cc_bitwise_or_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
