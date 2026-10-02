from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MaxRoiPool(_OperatorBenchmark):
    operator = "MaxRoiPool"
    case_name = "test_cc_maxroipool_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
