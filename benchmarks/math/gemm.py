from benchmarks._operator import OperatorBenchmark


class Gemm(OperatorBenchmark):
    operator = "Gemm"
    case_name = "test_cc_gemm_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
