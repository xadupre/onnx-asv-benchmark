from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitShift(_OperatorBenchmark):
    operator = "BitShift"
    case_name = "test_cc_bitshift_right_u8_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
