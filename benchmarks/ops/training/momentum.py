from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Momentum(_OperatorBenchmark):
    operator = "Momentum"
    case_name = "test_momentum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
