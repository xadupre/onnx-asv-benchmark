from benchmarks._operator import OperatorBenchmark


class BitwiseXor(OperatorBenchmark):
    operator = "BitwiseXor"
    case_name = "test_cc_bitwise_xor_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
