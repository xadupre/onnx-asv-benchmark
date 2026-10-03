from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class TreeEnsemble(_OperatorBenchmark):
    operator = "TreeEnsemble"
    case_name = "test_cc_treeensemble_single_tree_float_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'float16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
