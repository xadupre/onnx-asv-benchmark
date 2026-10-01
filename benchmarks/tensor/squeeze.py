from benchmarks._operator import OperatorBenchmark


class Squeeze(OperatorBenchmark):
    operator = "Squeeze"
    case_name = "test_cc_squeeze_axes_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
