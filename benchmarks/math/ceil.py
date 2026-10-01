from benchmarks._operator import OperatorBenchmark


class Ceil(OperatorBenchmark):
    operator = "Ceil"
    case_name = "test_cc_ceil_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
