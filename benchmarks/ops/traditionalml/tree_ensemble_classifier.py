from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class TreeEnsembleClassifier(_OperatorBenchmark):
    operator = "TreeEnsembleClassifier"
    case_name = "test_cc_treeensembleclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
