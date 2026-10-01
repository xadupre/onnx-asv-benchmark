from benchmarks._operator import OperatorBenchmark


class RandomNormalLike(OperatorBenchmark):
    operator = "RandomNormalLike"
    case_name = "test_cc_randomnormallike_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
