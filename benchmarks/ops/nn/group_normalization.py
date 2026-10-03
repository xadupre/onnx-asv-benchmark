from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GroupNormalization(_OperatorBenchmark):
    operator = "GroupNormalization"
    case_name = "test_cc_group_normalization_example_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
