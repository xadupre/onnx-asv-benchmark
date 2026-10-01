from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Transpose(_OperatorBenchmark):
    operator = "Transpose"
    case_name = "test_cc_transpose_default_perm_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
