from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BlackmanWindow(_OperatorBenchmark):
    operator = "BlackmanWindow"
    case_name = "test_cc_blackmanwindow_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
