from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Shape(_OperatorBenchmark):
    operator = "Shape"
    case_name = "test_cc_shape_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
