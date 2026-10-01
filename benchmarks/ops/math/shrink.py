from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Shrink(_OperatorBenchmark):
    operator = "Shrink"
    case_name = "test_cc_shrink_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
