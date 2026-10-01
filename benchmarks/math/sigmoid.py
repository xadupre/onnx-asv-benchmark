from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sigmoid(_OperatorBenchmark):
    operator = "Sigmoid"
    case_name = "test_cc_sigmoid_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
