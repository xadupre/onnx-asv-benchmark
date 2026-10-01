from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceLength(_OperatorBenchmark):
    operator = "SequenceLength"
    case_name = "test_cc_sequence_length_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
