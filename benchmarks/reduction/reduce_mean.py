from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceMean(_OperatorBenchmark):
    operator = "ReduceMean"
    case_name = "test_cc_reducemean_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
