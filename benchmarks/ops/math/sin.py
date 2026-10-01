from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sin(_OperatorBenchmark):
    operator = "Sin"
    case_name = "test_cc_sin_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
