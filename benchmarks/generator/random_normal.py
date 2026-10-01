from benchmarks._operator import OperatorBenchmark


class RandomNormal(OperatorBenchmark):
    operator = "RandomNormal"
    case_name = "test_cc_randomnormal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
