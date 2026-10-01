from benchmarks._operator import OperatorBenchmark


class Multinomial(OperatorBenchmark):
    operator = "Multinomial"
    case_name = "test_cc_multinomial_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
