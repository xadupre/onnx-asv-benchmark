from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LeakyRelu(_OperatorBenchmark):
    operator = "LeakyRelu"
    case_name = "test_cc_leakyrelu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
