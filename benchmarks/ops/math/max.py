from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Max(_OperatorBenchmark):
    operator = "Max"
    case_name = "test_cc_max_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
