from benchmarks._operator import OperatorBenchmark


class Einsum(OperatorBenchmark):
    operator = "Einsum"
    case_name = "test_cc_einsum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
