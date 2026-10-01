from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Acos(_OperatorBenchmark):
    operator = "Acos"
    case_name = "test_cc_acos_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
