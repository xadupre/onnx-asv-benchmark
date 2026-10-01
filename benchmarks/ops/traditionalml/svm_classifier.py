from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SVMClassifier(_OperatorBenchmark):
    operator = "SVMClassifier"
    case_name = "test_cc_svmclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
