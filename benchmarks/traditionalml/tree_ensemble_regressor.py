from benchmarks._operator import OperatorBenchmark


class TreeEnsembleRegressor(OperatorBenchmark):
    operator = "TreeEnsembleRegressor"
    case_name = "test_cc_treeensembleregressor_sum_single_target_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
