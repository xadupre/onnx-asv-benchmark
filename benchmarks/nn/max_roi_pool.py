from benchmarks._operator import OperatorBenchmark


class MaxRoiPool(OperatorBenchmark):
    operator = "MaxRoiPool"
    case_name = "test_cc_maxroipool_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
