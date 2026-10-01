from benchmarks._operator import OperatorBenchmark


class ScatterElements(OperatorBenchmark):
    operator = "ScatterElements"
    case_name = "test_cc_scatter_elements_without_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
