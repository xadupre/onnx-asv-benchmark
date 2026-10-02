from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SwiGLU(_OperatorBenchmark):
    operator = "SwiGLU"
    case_name = "test_cc_swiglu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
