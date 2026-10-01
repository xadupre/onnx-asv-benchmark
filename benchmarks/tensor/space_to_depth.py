from benchmarks._operator import OperatorBenchmark


class SpaceToDepth(OperatorBenchmark):
    operator = "SpaceToDepth"
    case_name = "test_cc_spacetodepth_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
