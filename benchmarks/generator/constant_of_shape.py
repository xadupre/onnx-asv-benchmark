from benchmarks._operator import OperatorBenchmark


class ConstantOfShape(OperatorBenchmark):
    operator = "ConstantOfShape"
    case_name = "test_constantofshape_float_ones_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
