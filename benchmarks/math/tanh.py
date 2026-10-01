from benchmarks._operator import OperatorBenchmark


class Tanh(OperatorBenchmark):
    operator = "Tanh"
    case_name = "test_cc_tanh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
