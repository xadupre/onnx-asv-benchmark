from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Tan(_OperatorBenchmark):
    operator = "Tan"
    case_name = "test_cc_tan_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
