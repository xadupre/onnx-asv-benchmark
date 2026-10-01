from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class QLinearConv(_OperatorBenchmark):
    operator = "QLinearConv"
    case_name = "test_cc_qlinearconv_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
