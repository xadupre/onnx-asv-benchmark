from benchmarks._operator import OperatorBenchmark


class SequenceEmpty(OperatorBenchmark):
    operator = "SequenceEmpty"
    case_name = "test_cc_sequence_empty_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
