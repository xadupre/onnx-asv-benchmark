from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceConstruct(_OperatorBenchmark):
    operator = "SequenceConstruct"
    case_name = "test_cc_sequence_construct_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
