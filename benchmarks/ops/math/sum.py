from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sum(_OperatorBenchmark):
    operator = "Sum"
    case_name = "test_cc_sum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
