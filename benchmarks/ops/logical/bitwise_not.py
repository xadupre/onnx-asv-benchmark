from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitwiseNot(_OperatorBenchmark):
    operator = "BitwiseNot"
    case_name = "test_cc_bitwise_not_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
