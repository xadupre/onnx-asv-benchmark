from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ArgMin(_OperatorBenchmark):
    operator = "ArgMin"
    case_name = "test_cc_argmin_no_keepdims_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
