from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ConstantOfShape(_OperatorBenchmark):
    operator = "ConstantOfShape"
    case_name = "test_constantofshape_float_ones_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
