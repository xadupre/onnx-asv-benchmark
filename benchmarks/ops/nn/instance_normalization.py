from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class InstanceNormalization(_OperatorBenchmark):
    operator = "InstanceNormalization"
    case_name = "test_cc_instancenorm_example_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
