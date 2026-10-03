from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class STFT(_OperatorBenchmark):
    operator = "STFT"
    case_name = "test_cc_stft_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
