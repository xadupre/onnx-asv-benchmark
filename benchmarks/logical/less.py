from benchmarks._operator import OperatorBenchmark


class Less(OperatorBenchmark):
    operator = "Less"
    case_name = "test_cc_less_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
