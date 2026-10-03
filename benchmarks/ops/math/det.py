from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Det(_OperatorBenchmark):
    operator = "Det"
    case_name = "test_cc_det_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
