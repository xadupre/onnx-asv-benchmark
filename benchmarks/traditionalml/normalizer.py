from benchmarks._operator import OperatorBenchmark


class Normalizer(OperatorBenchmark):
    operator = "Normalizer"
    case_name = "test_cc_normalizer_l2_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
