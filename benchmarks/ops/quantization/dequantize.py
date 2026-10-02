from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Dequantize(_OperatorBenchmark):
    operator = "Dequantize"
    case_name = "test_cc_dequantize_int4"
    case_mode = "TEST"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
