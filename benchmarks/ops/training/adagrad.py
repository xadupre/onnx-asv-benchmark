from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Adagrad(_OperatorBenchmark):
    operator = "Adagrad"
    case_name = "test_adagrad_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
