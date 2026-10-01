from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitwiseOr(_OperatorBenchmark):
    operator = "BitwiseOr"
    case_name = "test_cc_bitwise_or_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
