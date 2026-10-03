from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class TreeEnsembleClassifier(_OperatorBenchmark):
    operator = "TreeEnsembleClassifier"
    case_name = "test_cc_treeensembleclassifier_int64_binary_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'int64', 'int32')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
