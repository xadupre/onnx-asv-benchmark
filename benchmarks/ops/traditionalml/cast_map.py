from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CastMap(_OperatorBenchmark):
    operator = "CastMap"
    case_name = "test_cc_cast_map_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
