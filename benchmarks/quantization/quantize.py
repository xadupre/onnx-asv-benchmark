from benchmarks._operator import OperatorBenchmark


class Quantize(OperatorBenchmark):
    operator = "Quantize"
    case_name = "test_cc_quantize_int4"
    case_mode = "TEST"
    backends = ("onnx-light",)
