from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceSumSquare(_OperatorBenchmark):
    operator = "ReduceSumSquare"
    case_name = "test_cc_reducesumsquare_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
