from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Multinomial(_OperatorBenchmark):
    operator = "Multinomial"
    case_name = "test_cc_multinomial_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
