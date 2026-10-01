from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Mish(_OperatorBenchmark):
    operator = "Mish"
    case_name = "test_cc_mish_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
