from benchmarks._operator import OperatorBenchmark


class HammingWindow(OperatorBenchmark):
    operator = "HammingWindow"
    case_name = "test_cc_hammingwindow_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
