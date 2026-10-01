from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class StringNormalizer(_OperatorBenchmark):
    operator = "StringNormalizer"
    case_name = "test_cc_string_normalizer_lower_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
