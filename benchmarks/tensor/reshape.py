from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Reshape(_OperatorBenchmark):
    operator = "Reshape"
    case_name = "test_cc_reshape_reordered_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
