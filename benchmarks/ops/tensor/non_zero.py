from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class NonZero(_OperatorBenchmark):
    operator = "NonZero"
    case_name = "test_cc_nonzero_2d_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
