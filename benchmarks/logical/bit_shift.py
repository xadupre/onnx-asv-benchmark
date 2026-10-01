from benchmarks._operator import OperatorBenchmark


class BitShift(OperatorBenchmark):
    operator = "BitShift"
    case_name = "test_cc_bitshift_right_u8_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
