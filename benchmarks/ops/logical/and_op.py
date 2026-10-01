from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class And(_OperatorBenchmark):
    operator = "And"
    case_name = "test_cc_and_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
