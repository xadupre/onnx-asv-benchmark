from benchmarks._operator import OperatorBenchmark


class ConcatFromSequence(OperatorBenchmark):
    operator = "ConcatFromSequence"
    case_name = "test_cc_concat_from_sequence_axis_0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
