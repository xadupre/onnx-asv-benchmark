from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class HardSigmoid(_OperatorBenchmark):
    operator = "HardSigmoid"
    case_name = "test_cc_hardsigmoid_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
