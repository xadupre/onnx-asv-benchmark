from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Hardmax(_OperatorBenchmark):
    operator = "Hardmax"
    case_name = "test_cc_hardmax_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
