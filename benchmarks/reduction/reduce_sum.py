from benchmarks._operator import OperatorBenchmark


class ReduceSum(OperatorBenchmark):
    operator = "ReduceSum"
    case_name = "test_cc_reducesum_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
