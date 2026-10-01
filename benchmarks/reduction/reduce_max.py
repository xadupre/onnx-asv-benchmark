from benchmarks._operator import OperatorBenchmark


class ReduceMax(OperatorBenchmark):
    operator = "ReduceMax"
    case_name = "test_cc_reducemax_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
