from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceErase(_OperatorBenchmark):
    operator = "SequenceErase"
    case_name = "test_cc_sequence_erase_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
