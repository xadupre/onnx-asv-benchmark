from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class DynamicQuantizeLinear(_OperatorBenchmark):
    operator = "DynamicQuantizeLinear"
    case_name = "test_dynamicquantizelinear_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
