from benchmarks._operator import OperatorBenchmark


class Range(OperatorBenchmark):
    operator = "Range"
    case_name = "test_range_float_type_positive_delta_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
