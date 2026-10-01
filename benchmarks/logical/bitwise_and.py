from benchmarks._operator import OperatorBenchmark


class BitwiseAnd(OperatorBenchmark):
    operator = "BitwiseAnd"
    case_name = "test_cc_bitwise_and_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
