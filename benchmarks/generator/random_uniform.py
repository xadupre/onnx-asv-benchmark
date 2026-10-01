from benchmarks._operator import OperatorBenchmark


class RandomUniform(OperatorBenchmark):
    operator = "RandomUniform"
    case_name = "test_cc_randomuniform_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
