from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Unique(_OperatorBenchmark):
    operator = "Unique"
    case_name = "test_cc_unique_not_sorted_without_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
