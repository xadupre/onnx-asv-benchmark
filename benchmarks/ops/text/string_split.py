from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class StringSplit(_OperatorBenchmark):
    operator = "StringSplit"
    case_name = "test_cc_string_split_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
