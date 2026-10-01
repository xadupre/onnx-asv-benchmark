from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Scatter(_OperatorBenchmark):
    operator = "Scatter"
    case_name = "test_cc_scatter_without_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
