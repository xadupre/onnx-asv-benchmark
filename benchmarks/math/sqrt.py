from benchmarks._operator import OperatorBenchmark


class Sqrt(OperatorBenchmark):
    operator = "Sqrt"
    case_name = "test_cc_sqrt_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
