from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Range(_OperatorBenchmark):
    operator = "Range"
    case_name = "test_range_float_type_positive_delta_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'int16', 'int32', 'int64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
