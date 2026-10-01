from benchmarks._operator import OperatorBenchmark


class BitwiseNot(OperatorBenchmark):
    operator = "BitwiseNot"
    case_name = "test_cc_bitwise_not_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
