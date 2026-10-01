from benchmarks._operator import OperatorBenchmark


class Asinh(OperatorBenchmark):
    operator = "Asinh"
    case_name = "test_cc_asinh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
