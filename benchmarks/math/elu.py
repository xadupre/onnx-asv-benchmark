from benchmarks._operator import OperatorBenchmark


class Elu(OperatorBenchmark):
    operator = "Elu"
    case_name = "test_cc_elu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
