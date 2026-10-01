from benchmarks._operator import OperatorBenchmark


class LogSoftmax(OperatorBenchmark):
    operator = "LogSoftmax"
    case_name = "test_cc_logsoftmax_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
