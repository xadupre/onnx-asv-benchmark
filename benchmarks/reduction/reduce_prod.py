from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceProd(_OperatorBenchmark):
    operator = "ReduceProd"
    case_name = "test_cc_reduceprod_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
