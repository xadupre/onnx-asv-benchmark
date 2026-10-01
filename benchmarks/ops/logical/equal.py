from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Equal(_OperatorBenchmark):
    operator = "Equal"
    case_name = "test_cc_equal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
