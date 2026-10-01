from benchmarks._operator import OperatorBenchmark


class Log(OperatorBenchmark):
    operator = "Log"
    case_name = "test_cc_log_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
