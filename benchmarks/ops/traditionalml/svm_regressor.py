from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SVMRegressor(_OperatorBenchmark):
    operator = "SVMRegressor"
    case_name = "test_cc_svmregressor_linear_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'int64', 'int32')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
