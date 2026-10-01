from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Trilu(_OperatorBenchmark):
    operator = "Trilu"
    case_name = "test_cc_trilu_upper_default_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
