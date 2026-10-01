from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Mean(_OperatorBenchmark):
    operator = "Mean"
    case_name = "test_cc_mean_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
