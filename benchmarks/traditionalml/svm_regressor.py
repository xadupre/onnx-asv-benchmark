from benchmarks._operator import OperatorBenchmark


class SVMRegressor(OperatorBenchmark):
    operator = "SVMRegressor"
    case_name = "test_cc_svmregressor_linear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
