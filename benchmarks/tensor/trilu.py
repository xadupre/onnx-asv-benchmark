from benchmarks._operator import OperatorBenchmark


class Trilu(OperatorBenchmark):
    operator = "Trilu"
    case_name = "test_cc_trilu_upper_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
