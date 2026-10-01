from benchmarks._operator import OperatorBenchmark


class Imputer(OperatorBenchmark):
    operator = "Imputer"
    case_name = "test_cc_imputer_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
