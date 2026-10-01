from benchmarks._operator import OperatorBenchmark


class LpPool(OperatorBenchmark):
    operator = "LpPool"
    case_name = "test_cc_lppool_1d_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
