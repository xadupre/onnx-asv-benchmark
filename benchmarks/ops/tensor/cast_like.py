from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CastLike(_OperatorBenchmark):
    operator = "CastLike"
    case_name = "test_cc_castlike_FLOAT_to_DOUBLE_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
