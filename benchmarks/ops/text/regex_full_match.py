from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RegexFullMatch(_OperatorBenchmark):
    operator = "RegexFullMatch"
    case_name = "test_cc_regex_full_match_basic_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
