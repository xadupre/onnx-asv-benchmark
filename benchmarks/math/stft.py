from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class STFT(_OperatorBenchmark):
    operator = "STFT"
    case_name = "test_cc_stft_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
