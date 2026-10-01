from benchmarks._operator import OperatorBenchmark


class Asin(OperatorBenchmark):
    operator = "Asin"
    case_name = "test_cc_asin_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
