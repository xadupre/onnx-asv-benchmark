from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LinearRegressor(_OperatorBenchmark):
    operator = "LinearRegressor"
    case_name = "test_cc_linearregressor_single_target_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
