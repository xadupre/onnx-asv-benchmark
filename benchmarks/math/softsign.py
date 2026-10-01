from benchmarks._operator import OperatorBenchmark


class Softsign(OperatorBenchmark):
    operator = "Softsign"
    case_name = "test_cc_softsign_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
