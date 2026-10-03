from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LRN(_OperatorBenchmark):
    operator = "LRN"
    case_name = "test_cc_lrn_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
