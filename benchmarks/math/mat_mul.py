from benchmarks._operator import OperatorBenchmark


class MatMul(OperatorBenchmark):
    operator = "MatMul"
    case_name = "test_cc_matmul_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
