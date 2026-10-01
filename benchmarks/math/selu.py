from benchmarks._operator import OperatorBenchmark


class Selu(OperatorBenchmark):
    operator = "Selu"
    case_name = "test_cc_selu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
