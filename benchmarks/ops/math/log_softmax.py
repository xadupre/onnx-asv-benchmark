from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LogSoftmax(_OperatorBenchmark):
    operator = "LogSoftmax"
    case_name = "test_cc_logsoftmax_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
