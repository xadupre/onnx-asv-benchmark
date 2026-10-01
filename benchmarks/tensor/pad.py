from benchmarks._operator import OperatorBenchmark


class Pad(OperatorBenchmark):
    operator = "Pad"
    case_name = "test_cc_pad_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
