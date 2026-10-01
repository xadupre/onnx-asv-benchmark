from benchmarks._operator import OperatorBenchmark


class Relu(OperatorBenchmark):
    operator = "Relu"
    case_name = "test_cc_relu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
