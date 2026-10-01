from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Not(_OperatorBenchmark):
    operator = "Not"
    case_name = "test_cc_not_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
