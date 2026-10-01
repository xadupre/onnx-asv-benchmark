from benchmarks._operator import OperatorBenchmark


class Neg(OperatorBenchmark):
    operator = "Neg"
    case_name = "test_cc_neg_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
