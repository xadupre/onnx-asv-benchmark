from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Bernoulli(_OperatorBenchmark):
    operator = "Bernoulli"
    case_name = "test_cc_bernoulli_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
