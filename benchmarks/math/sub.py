from benchmarks._operator import OperatorBenchmark


class Sub(OperatorBenchmark):
    operator = "Sub"
    case_name = "test_cc_sub_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
