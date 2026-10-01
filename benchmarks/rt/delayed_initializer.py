from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class DelayedInitializer(_OperatorBenchmark):
    operator = "DelayedInitializer"
    case_name = "test_cc_delayedinitializer_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
