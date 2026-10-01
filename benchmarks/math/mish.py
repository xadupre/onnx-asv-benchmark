from benchmarks._operator import OperatorBenchmark


class Mish(OperatorBenchmark):
    operator = "Mish"
    case_name = "test_cc_mish_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
