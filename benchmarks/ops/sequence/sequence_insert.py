from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SequenceInsert(_OperatorBenchmark):
    operator = "SequenceInsert"
    case_name = "test_cc_sequence_insert_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
