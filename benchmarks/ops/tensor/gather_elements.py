from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GatherElements(_OperatorBenchmark):
    operator = "GatherElements"
    case_name = "test_cc_gather_elements_0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
