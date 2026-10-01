from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LinearAttention(_OperatorBenchmark):
    operator = "LinearAttention"
    case_name = "test_cc_linear_attention_linear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
