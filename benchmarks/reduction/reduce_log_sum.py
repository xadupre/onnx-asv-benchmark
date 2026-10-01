from benchmarks._operator import OperatorBenchmark


class ReduceLogSum(OperatorBenchmark):
    operator = "ReduceLogSum"
    case_name = "test_cc_reducelogsum_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
