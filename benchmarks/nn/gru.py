from benchmarks._operator import OperatorBenchmark


class GRU(OperatorBenchmark):
    operator = "GRU"
    case_name = "test_cc_gru_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
