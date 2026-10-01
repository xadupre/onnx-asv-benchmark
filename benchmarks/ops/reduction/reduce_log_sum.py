from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceLogSum(_OperatorBenchmark):
    operator = "ReduceLogSum"
    case_name = "test_cc_reducelogsum_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
