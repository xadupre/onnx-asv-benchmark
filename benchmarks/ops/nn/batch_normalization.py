from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BatchNormalization(_OperatorBenchmark):
    operator = "BatchNormalization"
    case_name = "test_cc_batchnorm_example_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
