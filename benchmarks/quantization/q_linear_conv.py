from benchmarks._operator import OperatorBenchmark


class QLinearConv(OperatorBenchmark):
    operator = "QLinearConv"
    case_name = "test_cc_qlinearconv_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
