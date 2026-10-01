from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ConvTranspose(_OperatorBenchmark):
    operator = "ConvTranspose"
    case_name = "test_cc_convtranspose_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
