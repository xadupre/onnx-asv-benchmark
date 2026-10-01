from benchmarks._operator import OperatorBenchmark


class LinearAttention(OperatorBenchmark):
    operator = "LinearAttention"
    case_name = "test_cc_linear_attention_linear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
