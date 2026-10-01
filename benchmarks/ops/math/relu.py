from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Relu(_OperatorBenchmark):
    operator = "Relu"
    case_name = "test_cc_relu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
