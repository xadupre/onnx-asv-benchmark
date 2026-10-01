from benchmarks._operator import OperatorBenchmark


class Gelu(OperatorBenchmark):
    operator = "Gelu"
    case_name = "test_cc_gelu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
