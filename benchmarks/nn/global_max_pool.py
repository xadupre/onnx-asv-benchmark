from benchmarks._operator import OperatorBenchmark


class GlobalMaxPool(OperatorBenchmark):
    operator = "GlobalMaxPool"
    case_name = "test_cc_globalmaxpool_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
