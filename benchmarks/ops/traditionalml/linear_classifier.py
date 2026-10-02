from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LinearClassifier(_OperatorBenchmark):
    operator = "LinearClassifier"
    case_name = "test_cc_linearclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
