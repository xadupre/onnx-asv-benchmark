from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SpaceToDepth(_OperatorBenchmark):
    operator = "SpaceToDepth"
    case_name = "test_cc_spacetodepth_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
