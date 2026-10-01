from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Elu(_OperatorBenchmark):
    operator = "Elu"
    case_name = "test_cc_elu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
