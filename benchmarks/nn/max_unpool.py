from benchmarks._operator import OperatorBenchmark


class MaxUnpool(OperatorBenchmark):
    operator = "MaxUnpool"
    case_name = "test_cc_maxunpool_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
