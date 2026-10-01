from benchmarks._operator import OperatorBenchmark


class GroupNormalization(OperatorBenchmark):
    operator = "GroupNormalization"
    case_name = "test_cc_group_normalization_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
