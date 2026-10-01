from benchmarks._operator import OperatorBenchmark


class Mean(OperatorBenchmark):
    operator = "Mean"
    case_name = "test_cc_mean_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
