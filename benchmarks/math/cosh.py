from benchmarks._operator import OperatorBenchmark


class Cosh(OperatorBenchmark):
    operator = "Cosh"
    case_name = "test_cc_cosh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
