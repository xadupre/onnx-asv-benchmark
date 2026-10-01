from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SplitToSequence(_OperatorBenchmark):
    operator = "SplitToSequence"
    case_name = "test_cc_split_to_sequence_1_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
