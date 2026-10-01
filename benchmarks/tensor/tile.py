from benchmarks._operator import OperatorBenchmark


class Tile(OperatorBenchmark):
    operator = "Tile"
    case_name = "test_cc_tile_precomputed_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
