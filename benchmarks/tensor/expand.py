from benchmarks._operator import OperatorBenchmark


class Expand(OperatorBenchmark):
    operator = "Expand"
    case_name = "test_cc_expand_dim_changed_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
