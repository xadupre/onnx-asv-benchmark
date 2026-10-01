from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Greater(_OperatorBenchmark):
    operator = "Greater"
    case_name = "test_cc_greater_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
