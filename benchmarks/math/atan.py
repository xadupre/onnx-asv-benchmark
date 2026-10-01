from benchmarks._operator import OperatorBenchmark


class Atan(OperatorBenchmark):
    operator = "Atan"
    case_name = "test_cc_atan_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
