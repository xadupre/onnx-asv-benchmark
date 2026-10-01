from benchmarks._operator import OperatorBenchmark


class Erf(OperatorBenchmark):
    operator = "Erf"
    case_name = "test_cc_erf_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
