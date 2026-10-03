from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Attention(_OperatorBenchmark):
    operator = "Attention"
    case_name = "test_cc_attention_prefill_mha_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
