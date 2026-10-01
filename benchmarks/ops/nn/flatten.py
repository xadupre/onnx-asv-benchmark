from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Flatten(_OperatorBenchmark):
    operator = "Flatten"
    case_name = "test_cc_flatten_default_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
