from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class AveragePool(_OperatorBenchmark):
    operator = "AveragePool"
    case_name = "test_cc_averagepool_2d_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
