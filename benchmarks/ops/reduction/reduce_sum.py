from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceSum(_OperatorBenchmark):
    operator = "ReduceSum"
    case_name = "test_cc_reducesum_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
