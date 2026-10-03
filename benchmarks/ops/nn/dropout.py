from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Dropout(_OperatorBenchmark):
    operator = "Dropout"
    case_name = "test_cc_dropout_default_inference_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'bfloat16', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
