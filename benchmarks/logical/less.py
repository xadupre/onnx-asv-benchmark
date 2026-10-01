from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Less(_OperatorBenchmark):
    operator = "Less"
    case_name = "test_cc_less_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
