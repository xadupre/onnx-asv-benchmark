from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class DictVectorizer(_OperatorBenchmark):
    operator = "DictVectorizer"
    case_name = "test_cc_dict_vectorizer_string_int64"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
