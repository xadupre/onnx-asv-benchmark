from benchmarks._operator import OperatorBenchmark


class LRN(OperatorBenchmark):
    operator = "LRN"
    case_name = "test_cc_lrn_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
