from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ArrayFeatureExtractor(_OperatorBenchmark):
    operator = "ArrayFeatureExtractor"
    case_name = "test_ai_onnx_ml_array_feature_extractor_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
