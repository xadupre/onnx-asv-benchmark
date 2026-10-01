from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class EyeLike(_OperatorBenchmark):
    operator = "EyeLike"
    case_name = "test_eyelike_without_dtype_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
