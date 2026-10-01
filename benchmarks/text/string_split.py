from benchmarks._operator import OperatorBenchmark


class StringSplit(OperatorBenchmark):
    operator = "StringSplit"
    case_name = "test_cc_string_split_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
