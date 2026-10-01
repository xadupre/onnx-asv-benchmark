from benchmarks._operator import OperatorBenchmark


class ReduceSumSquare(OperatorBenchmark):
    operator = "ReduceSumSquare"
    case_name = "test_cc_reducesumsquare_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
