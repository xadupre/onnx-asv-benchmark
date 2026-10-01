from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ConcatFromSequence(_OperatorBenchmark):
    operator = "ConcatFromSequence"
    case_name = "test_cc_concat_from_sequence_axis_0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
