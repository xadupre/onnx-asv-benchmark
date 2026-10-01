from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Sub(_OperatorBenchmark):
    operator = "Sub"
    case_name = "test_cc_sub_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
