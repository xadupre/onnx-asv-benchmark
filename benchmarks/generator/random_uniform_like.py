from benchmarks._operator import OperatorBenchmark


class RandomUniformLike(OperatorBenchmark):
    operator = "RandomUniformLike"
    case_name = "test_cc_randomuniformlike_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
