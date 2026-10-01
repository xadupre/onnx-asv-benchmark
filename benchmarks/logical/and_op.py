from benchmarks._operator import OperatorBenchmark


class And(OperatorBenchmark):
    operator = "And"
    case_name = "test_cc_and_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
