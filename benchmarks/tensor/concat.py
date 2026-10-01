from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Concat(_OperatorBenchmark):
    operator = "Concat"
    case_name = "test_cc_concat_1d_axis_0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
