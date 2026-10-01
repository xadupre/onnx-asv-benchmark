from benchmarks._operator import OperatorBenchmark


class GlobalLpPool(OperatorBenchmark):
    operator = "GlobalLpPool"
    case_name = "test_cc_globallppool_lp1_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
