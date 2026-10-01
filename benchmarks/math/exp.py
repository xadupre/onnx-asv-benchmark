from benchmarks._operator import OperatorBenchmark


class Exp(OperatorBenchmark):
    operator = "Exp"
    case_name = "test_cc_exp_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
