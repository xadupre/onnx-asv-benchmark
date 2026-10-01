from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RoiAlign(_OperatorBenchmark):
    operator = "RoiAlign"
    case_name = "test_cc_roialign_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
