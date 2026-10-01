from benchmarks._operator import OperatorBenchmark


class HardSwish(OperatorBenchmark):
    operator = "HardSwish"
    case_name = "test_cc_hardswish_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
