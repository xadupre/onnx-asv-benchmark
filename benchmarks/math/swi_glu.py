from benchmarks._operator import OperatorBenchmark


class SwiGLU(OperatorBenchmark):
    operator = "SwiGLU"
    case_name = "test_cc_swiglu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
