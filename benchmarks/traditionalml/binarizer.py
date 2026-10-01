from benchmarks._operator import OperatorBenchmark


class Binarizer(OperatorBenchmark):
    operator = "Binarizer"
    case_name = "test_ai_onnx_ml_binarizer_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
