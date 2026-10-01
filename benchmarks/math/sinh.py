from benchmarks._operator import OperatorBenchmark


class Sinh(OperatorBenchmark):
    operator = "Sinh"
    case_name = "test_cc_sinh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
