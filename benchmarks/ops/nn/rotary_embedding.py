from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RotaryEmbedding(_OperatorBenchmark):
    operator = "RotaryEmbedding"
    case_name = "test_cc_rotary_embedding_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
