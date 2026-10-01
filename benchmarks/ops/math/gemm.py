from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Gemm(_OperatorBenchmark):
    operator = "Gemm"
    case_name = "test_cc_gemm_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
