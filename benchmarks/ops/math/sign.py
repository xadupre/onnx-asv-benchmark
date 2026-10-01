from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sign(_OperatorBenchmark):
    operator = "Sign"
    case_name = "test_cc_sign_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
