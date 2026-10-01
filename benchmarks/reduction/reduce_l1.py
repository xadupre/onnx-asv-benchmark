from benchmarks._operator import OperatorBenchmark


class ReduceL1(OperatorBenchmark):
    operator = "ReduceL1"
    case_name = "test_cc_reducel1_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
