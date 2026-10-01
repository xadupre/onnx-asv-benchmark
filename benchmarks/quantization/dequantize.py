from benchmarks._operator import OperatorBenchmark


class Dequantize(OperatorBenchmark):
    operator = "Dequantize"
    case_name = "test_cc_dequantize_int4"
    case_mode = "TEST"
    backends = ("onnx-light",)
