from benchmarks._operator import OperatorBenchmark


class Not(OperatorBenchmark):
    operator = "Not"
    case_name = "test_cc_not_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
