from benchmarks._operator import OperatorBenchmark


class TfIdfVectorizer(OperatorBenchmark):
    operator = "TfIdfVectorizer"
    case_name = "test_cc_tfidfvectorizer_tf_only_bigrams_skip0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
