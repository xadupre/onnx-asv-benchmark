from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Mod(_OperatorBenchmark):
    operator = "Mod"
    case_name = "test_cc_mod_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
