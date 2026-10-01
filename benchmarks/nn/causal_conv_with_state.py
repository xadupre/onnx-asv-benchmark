from benchmarks._operator import OperatorBenchmark


class CausalConvWithState(OperatorBenchmark):
    operator = "CausalConvWithState"
    case_name = "test_cc_causal_conv_with_state_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
