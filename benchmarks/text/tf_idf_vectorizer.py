from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class TfIdfVectorizer(_OperatorBenchmark):
    operator = "TfIdfVectorizer"
    case_name = "test_cc_tfidfvectorizer_tf_only_bigrams_skip0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
