from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class PRelu(_OperatorBenchmark):
    operator = "PRelu"
    case_name = "test_cc_prelu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
