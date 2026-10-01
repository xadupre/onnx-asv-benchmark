from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Neg(_OperatorBenchmark):
    operator = "Neg"
    case_name = "test_cc_neg_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
