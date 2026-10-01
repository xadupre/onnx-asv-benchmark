from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Or(_OperatorBenchmark):
    operator = "Or"
    case_name = "test_cc_or_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
