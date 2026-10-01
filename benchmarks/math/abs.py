from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Abs(_OperatorBenchmark):
    operator = "Abs"
    case_name = "test_cc_abs_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
