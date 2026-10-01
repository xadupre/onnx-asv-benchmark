from benchmarks._operator import OperatorBenchmark


class TreeEnsembleClassifier(OperatorBenchmark):
    operator = "TreeEnsembleClassifier"
    case_name = "test_cc_treeensembleclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
