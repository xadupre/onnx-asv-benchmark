from benchmarks._operator import OperatorBenchmark


class LinearClassifier(OperatorBenchmark):
    operator = "LinearClassifier"
    case_name = "test_cc_linearclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
