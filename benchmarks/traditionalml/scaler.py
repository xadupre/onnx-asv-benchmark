from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Scaler(_OperatorBenchmark):
    operator = "Scaler"
    case_name = "test_cc_scaler_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
