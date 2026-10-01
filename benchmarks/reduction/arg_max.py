from benchmarks._operator import OperatorBenchmark


class ArgMax(OperatorBenchmark):
    operator = "ArgMax"
    case_name = "test_cc_argmax_no_keepdims_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
