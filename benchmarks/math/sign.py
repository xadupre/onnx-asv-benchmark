from benchmarks._operator import OperatorBenchmark


class Sign(OperatorBenchmark):
    operator = "Sign"
    case_name = "test_cc_sign_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
