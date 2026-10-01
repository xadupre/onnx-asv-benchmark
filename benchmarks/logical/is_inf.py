from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class IsInf(_OperatorBenchmark):
    operator = "IsInf"
    case_name = "test_cc_isinf_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
