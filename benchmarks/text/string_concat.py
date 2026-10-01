from benchmarks._operator import OperatorBenchmark


class StringConcat(OperatorBenchmark):
    operator = "StringConcat"
    case_name = "test_cc_string_concat_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
