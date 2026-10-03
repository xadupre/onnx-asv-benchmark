from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Binarizer(_OperatorBenchmark):
    operator = "Binarizer"
    case_name = "test_ai_onnx_ml_binarizer_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64', 'int64', 'int32')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
