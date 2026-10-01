from benchmarks._operator import OperatorBenchmark


class Tan(OperatorBenchmark):
    operator = "Tan"
    case_name = "test_cc_tan_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
