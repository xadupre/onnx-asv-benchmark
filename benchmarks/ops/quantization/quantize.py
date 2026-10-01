from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Quantize(_OperatorBenchmark):
    operator = "Quantize"
    case_name = "test_cc_quantize_int4"
    case_mode = "TEST"
    backends = ("onnx-light",)
