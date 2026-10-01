from benchmarks._operator import OperatorBenchmark


class Atanh(OperatorBenchmark):
    operator = "Atanh"
    case_name = "test_cc_atanh_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
