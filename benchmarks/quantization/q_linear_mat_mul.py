from benchmarks._operator import OperatorBenchmark


class QLinearMatMul(OperatorBenchmark):
    operator = "QLinearMatMul"
    case_name = "test_cc_qlinearmatmul_2D_uint8_float32_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
