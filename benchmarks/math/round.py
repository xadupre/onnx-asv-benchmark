from benchmarks._operator import OperatorBenchmark


class Round(OperatorBenchmark):
    operator = "Round"
    case_name = "test_cc_round_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
