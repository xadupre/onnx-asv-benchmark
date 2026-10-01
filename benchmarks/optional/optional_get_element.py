from benchmarks._operator import OperatorBenchmark


class OptionalGetElement(OperatorBenchmark):
    operator = "OptionalGetElement"
    case_name = "test_cc_optional_get_element_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
