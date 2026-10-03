from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LinearAttention(_OperatorBenchmark):
    operator = "LinearAttention"
    case_name = "test_cc_linear_attention_linear_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
