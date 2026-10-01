from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceMin(_OperatorBenchmark):
    operator = "ReduceMin"
    case_name = "test_cc_reducemin_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
