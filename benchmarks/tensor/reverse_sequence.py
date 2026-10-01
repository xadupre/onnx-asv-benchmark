from benchmarks._operator import OperatorBenchmark


class ReverseSequence(OperatorBenchmark):
    operator = "ReverseSequence"
    case_name = "test_cc_reversesequence_time_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
