from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class HammingWindow(_OperatorBenchmark):
    operator = "HammingWindow"
    case_name = "test_cc_hammingwindow_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int32', 'int64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
