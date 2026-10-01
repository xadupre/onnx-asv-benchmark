from benchmarks._operator import OperatorBenchmark


class Momentum(OperatorBenchmark):
    operator = "Momentum"
    case_name = "test_momentum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
