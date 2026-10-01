from benchmarks._operator import OperatorBenchmark


class HannWindow(OperatorBenchmark):
    operator = "HannWindow"
    case_name = "test_cc_hannwindow_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
