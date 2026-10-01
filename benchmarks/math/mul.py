from benchmarks._operator import OperatorBenchmark


class Mul(OperatorBenchmark):
    operator = "Mul"
    case_name = "test_cc_mul_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
