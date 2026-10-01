from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class FlexAttention(_OperatorBenchmark):
    operator = "FlexAttention"
    case_name = "test_cc_flexattention_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
