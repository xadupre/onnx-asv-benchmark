from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitwiseAnd(_OperatorBenchmark):
    operator = "BitwiseAnd"
    case_name = "test_cc_bitwise_and_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
