from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MaxPool(_OperatorBenchmark):
    operator = "MaxPool"
    case_name = "test_cc_maxpool_1d_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
