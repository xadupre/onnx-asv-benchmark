from benchmarks._operator import OperatorBenchmark


class ReduceLogSumExp(OperatorBenchmark):
    operator = "ReduceLogSumExp"
    case_name = "test_cc_reducelogsumexp_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
