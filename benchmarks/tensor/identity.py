from benchmarks._operator import OperatorBenchmark


class Identity(OperatorBenchmark):
    operator = "Identity"
    case_name = "test_cc_identity_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
