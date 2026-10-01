from benchmarks._operator import OperatorBenchmark


class Swish(OperatorBenchmark):
    operator = "Swish"
    case_name = "test_cc_swish_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
