from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RandomNormalLike(_OperatorBenchmark):
    operator = "RandomNormalLike"
    case_name = "test_cc_randomnormallike_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
