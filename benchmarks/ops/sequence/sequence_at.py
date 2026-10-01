from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceAt(_OperatorBenchmark):
    operator = "SequenceAt"
    case_name = "test_cc_sequence_at_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
