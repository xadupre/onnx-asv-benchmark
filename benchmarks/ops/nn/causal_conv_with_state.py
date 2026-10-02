from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CausalConvWithState(_OperatorBenchmark):
    operator = "CausalConvWithState"
    case_name = "test_cc_causal_conv_with_state_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
