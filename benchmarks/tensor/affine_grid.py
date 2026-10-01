from benchmarks._operator import OperatorBenchmark


class AffineGrid(OperatorBenchmark):
    operator = "AffineGrid"
    case_name = "test_affine_grid_2d_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
