from benchmarks._operator import OperatorBenchmark


class LeakyRelu(OperatorBenchmark):
    operator = "LeakyRelu"
    case_name = "test_cc_leakyrelu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
