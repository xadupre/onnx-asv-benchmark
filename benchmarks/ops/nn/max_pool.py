from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MaxPool(_OperatorBenchmark):
    operator = "MaxPool"
    case_name = "test_cc_maxpool_1d_default_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64', 'int8', 'uint8')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
