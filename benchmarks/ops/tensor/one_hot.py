from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class OneHot(_OperatorBenchmark):
    operator = "OneHot"
    case_name = "test_onehot_without_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
