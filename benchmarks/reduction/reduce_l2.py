from benchmarks._operator import OperatorBenchmark


class ReduceL2(OperatorBenchmark):
    operator = "ReduceL2"
    case_name = "test_cc_reducel2_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
