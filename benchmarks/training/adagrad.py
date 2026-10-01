from benchmarks._operator import OperatorBenchmark


class Adagrad(OperatorBenchmark):
    operator = "Adagrad"
    case_name = "test_adagrad_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
