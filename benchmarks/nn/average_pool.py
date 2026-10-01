from benchmarks._operator import OperatorBenchmark


class AveragePool(OperatorBenchmark):
    operator = "AveragePool"
    case_name = "test_cc_averagepool_2d_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
