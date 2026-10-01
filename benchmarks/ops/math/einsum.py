from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Einsum(_OperatorBenchmark):
    operator = "Einsum"
    case_name = "test_cc_einsum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
