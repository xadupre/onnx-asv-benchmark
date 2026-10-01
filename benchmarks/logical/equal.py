from benchmarks._operator import OperatorBenchmark


class Equal(OperatorBenchmark):
    operator = "Equal"
    case_name = "test_cc_equal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
