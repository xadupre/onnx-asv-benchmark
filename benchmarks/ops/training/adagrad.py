from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Adagrad(_OperatorBenchmark):
    operator = "Adagrad"
    case_name = "test_adagrad_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
