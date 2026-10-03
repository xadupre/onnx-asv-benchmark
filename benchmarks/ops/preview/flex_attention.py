from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class FlexAttention(_OperatorBenchmark):
    operator = "FlexAttention"
    case_name = "test_cc_flexattention_basic_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
