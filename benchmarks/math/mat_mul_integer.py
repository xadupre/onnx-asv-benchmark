from benchmarks._operator import OperatorBenchmark


class MatMulInteger(OperatorBenchmark):
    operator = "MatMulInteger"
    case_name = "test_cc_matmulinteger_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
