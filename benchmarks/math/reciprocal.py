from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Reciprocal(_OperatorBenchmark):
    operator = "Reciprocal"
    case_name = "test_cc_reciprocal_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
