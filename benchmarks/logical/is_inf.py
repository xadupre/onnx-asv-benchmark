from benchmarks._operator import OperatorBenchmark


class IsInf(OperatorBenchmark):
    operator = "IsInf"
    case_name = "test_cc_isinf_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
