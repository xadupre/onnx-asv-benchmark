from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Swish(_OperatorBenchmark):
    operator = "Swish"
    case_name = "test_cc_swish_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'bfloat16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
