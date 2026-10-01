from benchmarks._operator import OperatorBenchmark


class SequenceInsert(OperatorBenchmark):
    operator = "SequenceInsert"
    case_name = "test_cc_sequence_insert_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
