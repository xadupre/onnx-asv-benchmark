from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GlobalMaxPool(_OperatorBenchmark):
    operator = "GlobalMaxPool"
    case_name = "test_cc_globalmaxpool_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
