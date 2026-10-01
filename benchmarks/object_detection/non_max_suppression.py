from benchmarks._operator import OperatorBenchmark


class NonMaxSuppression(OperatorBenchmark):
    operator = "NonMaxSuppression"
    case_name = "test_cc_nonmaxsuppression_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
