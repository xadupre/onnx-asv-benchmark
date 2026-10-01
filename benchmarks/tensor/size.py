from benchmarks._operator import OperatorBenchmark


class Size(OperatorBenchmark):
    operator = "Size"
    case_name = "test_cc_size_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
