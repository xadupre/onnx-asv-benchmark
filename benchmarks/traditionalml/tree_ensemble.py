from benchmarks._operator import OperatorBenchmark


class TreeEnsemble(OperatorBenchmark):
    operator = "TreeEnsemble"
    case_name = "test_cc_treeensemble_single_tree_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
