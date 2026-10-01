from benchmarks._operator import OperatorBenchmark


class DequantizeLinear(OperatorBenchmark):
    operator = "DequantizeLinear"
    case_name = "test_cc_dequantizelinear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
