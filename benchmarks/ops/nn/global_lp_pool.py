from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GlobalLpPool(_OperatorBenchmark):
    operator = "GlobalLpPool"
    case_name = "test_cc_globallppool_lp1_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
