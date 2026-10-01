from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class QuantizeLinear(_OperatorBenchmark):
    operator = "QuantizeLinear"
    case_name = "test_cc_quantizelinear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
