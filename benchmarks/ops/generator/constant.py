from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Constant(_OperatorBenchmark):
    operator = "Constant"
    case_name = "test_cc_constant_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
