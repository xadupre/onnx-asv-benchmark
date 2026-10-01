from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceL2(_OperatorBenchmark):
    operator = "ReduceL2"
    case_name = "test_cc_reducel2_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
