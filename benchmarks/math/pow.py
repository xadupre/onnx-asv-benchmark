from benchmarks._operator import OperatorBenchmark


class Pow(OperatorBenchmark):
    operator = "Pow"
    case_name = "test_cc_pow_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
