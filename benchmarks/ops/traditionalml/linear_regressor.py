from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LinearRegressor(_OperatorBenchmark):
    operator = "LinearRegressor"
    case_name = "test_cc_linearregressor_single_target_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'int64', 'int32')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
