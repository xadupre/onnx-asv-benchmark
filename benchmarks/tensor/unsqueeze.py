from benchmarks._operator import OperatorBenchmark


class Unsqueeze(OperatorBenchmark):
    operator = "Unsqueeze"
    case_name = "test_cc_unsqueeze_axes_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
