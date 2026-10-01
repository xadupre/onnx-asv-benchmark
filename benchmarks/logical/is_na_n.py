from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class IsNaN(_OperatorBenchmark):
    operator = "IsNaN"
    case_name = "test_cc_isnan_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
