from benchmarks._operator import OperatorBenchmark


class Optional(OperatorBenchmark):
    operator = "Optional"
    case_name = "test_cc_optional_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
