from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceMax(_OperatorBenchmark):
    operator = "ReduceMax"
    case_name = "test_cc_reducemax_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
