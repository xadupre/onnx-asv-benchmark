from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GatherND(_OperatorBenchmark):
    operator = "GatherND"
    case_name = "test_cc_gathernd_example_int32_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
