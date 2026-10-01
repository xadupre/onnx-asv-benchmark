from benchmarks._operator import OperatorBenchmark


class Acosh(OperatorBenchmark):
    operator = "Acosh"
    case_name = "test_cc_acosh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
