from benchmarks._operator import OperatorBenchmark


class FlexAttention(OperatorBenchmark):
    operator = "FlexAttention"
    case_name = "test_cc_flexattention_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
