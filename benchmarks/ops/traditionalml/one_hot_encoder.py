from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class OneHotEncoder(_OperatorBenchmark):
    operator = "OneHotEncoder"
    case_name = "test_cc_one_hot_encoder_int64_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int64', 'int32', 'float32', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
