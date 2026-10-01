from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Softmax(_OperatorBenchmark):
    operator = "Softmax"
    case_name = "test_cc_softmax_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
