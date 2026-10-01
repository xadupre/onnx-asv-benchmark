from benchmarks._operator import OperatorBenchmark


class ReduceMean(OperatorBenchmark):
    operator = "ReduceMean"
    case_name = "test_cc_reducemean_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
