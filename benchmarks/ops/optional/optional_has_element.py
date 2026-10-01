from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class OptionalHasElement(_OperatorBenchmark):
    operator = "OptionalHasElement"
    case_name = "test_cc_optional_has_element_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
