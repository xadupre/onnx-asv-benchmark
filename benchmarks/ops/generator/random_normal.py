from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RandomNormal(_OperatorBenchmark):
    operator = "RandomNormal"
    case_name = "test_cc_randomnormal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
