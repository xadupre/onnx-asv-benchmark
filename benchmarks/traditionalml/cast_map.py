from benchmarks._operator import OperatorBenchmark


class CastMap(OperatorBenchmark):
    operator = "CastMap"
    case_name = "test_cc_cast_map_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
