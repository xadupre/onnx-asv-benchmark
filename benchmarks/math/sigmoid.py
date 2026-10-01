from benchmarks._operator import OperatorBenchmark


class Sigmoid(OperatorBenchmark):
    operator = "Sigmoid"
    case_name = "test_cc_sigmoid_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
