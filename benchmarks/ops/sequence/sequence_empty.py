from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceEmpty(_OperatorBenchmark):
    operator = "SequenceEmpty"
    case_name = "test_cc_sequence_empty_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
