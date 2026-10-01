from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Xor(_OperatorBenchmark):
    operator = "Xor"
    case_name = "test_cc_xor_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
