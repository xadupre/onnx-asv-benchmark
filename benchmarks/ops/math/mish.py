from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Mish(_OperatorBenchmark):
    operator = "Mish"
    case_name = "test_cc_mish_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
