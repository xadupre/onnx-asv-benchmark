from benchmarks._operator import OperatorBenchmark


class ConvTranspose(OperatorBenchmark):
    operator = "ConvTranspose"
    case_name = "test_cc_convtranspose_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
