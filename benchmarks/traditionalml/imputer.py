from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Imputer(_OperatorBenchmark):
    operator = "Imputer"
    case_name = "test_cc_imputer_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
