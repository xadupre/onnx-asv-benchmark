from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RotaryEmbedding(_OperatorBenchmark):
    operator = "RotaryEmbedding"
    case_name = "test_cc_rotary_embedding_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
