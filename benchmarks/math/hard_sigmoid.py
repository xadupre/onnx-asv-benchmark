from benchmarks._operator import OperatorBenchmark


class HardSigmoid(OperatorBenchmark):
    operator = "HardSigmoid"
    case_name = "test_cc_hardsigmoid_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
