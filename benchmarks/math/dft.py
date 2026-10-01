from benchmarks._operator import OperatorBenchmark


class DFT(OperatorBenchmark):
    operator = "DFT"
    case_name = "test_cc_dft_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
