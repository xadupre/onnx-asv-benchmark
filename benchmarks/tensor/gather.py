from benchmarks._operator import OperatorBenchmark


class Gather(OperatorBenchmark):
    operator = "Gather"
    case_name = "test_cc_gather_0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
