from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Adam(_OperatorBenchmark):
    operator = "Adam"
    case_name = "test_cc_adam_single_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
