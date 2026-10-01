from benchmarks._operator import OperatorBenchmark


class DynamicQuantizeLinear(OperatorBenchmark):
    operator = "DynamicQuantizeLinear"
    case_name = "test_dynamicquantizelinear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
