from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RandomUniformLike(_OperatorBenchmark):
    operator = "RandomUniformLike"
    case_name = "test_cc_randomuniformlike_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
