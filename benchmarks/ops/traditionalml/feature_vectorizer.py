from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class FeatureVectorizer(_OperatorBenchmark):
    operator = "FeatureVectorizer"
    case_name = "test_cc_feature_vectorizer_two_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
