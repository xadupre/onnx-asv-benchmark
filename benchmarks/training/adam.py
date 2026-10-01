from benchmarks._operator import OperatorBenchmark


class Adam(OperatorBenchmark):
    operator = "Adam"
    case_name = "test_cc_adam_single_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
