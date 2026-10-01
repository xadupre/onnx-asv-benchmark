from benchmarks._operator import OperatorBenchmark


class Mod(OperatorBenchmark):
    operator = "Mod"
    case_name = "test_cc_mod_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
